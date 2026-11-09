"""Incident evidence generation: annotated snapshots, pre+post MP4 clips, and atomic JSON metadata."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import shutil
import threading
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon

from config_manager import AmrConfig, CorridorGeometry, StorageConfig
from risk_evaluator import RiskAssessmentResult, RiskLevel
from track_manager import TrackedObject

logger = logging.getLogger(__name__)


@dataclass
class PendingIncidentJob:
    """Tracks an in-flight incident awaiting post-event frames before encoding."""

    event_id: str
    folder_name: str
    incident_dir: Path
    track_snapshot: TrackedObject
    risk_result: RiskAssessmentResult
    frame_id: int
    iso_timestamp: str
    landing_frame: np.ndarray
    pre_event_frames: List[np.ndarray]
    post_event_frames: List[np.ndarray] = field(default_factory=list)
    remaining_post_frames: int = 0
    is_preexisting_debris: bool = False


@dataclass(frozen=True)
class CompletedIncidentRecord:
    """Verified evidence artifact paths and status returned to main.py."""

    event_id: str
    track_id: int
    incident_dir: Path
    snapshot_path: Path
    video_path: Path
    metadata_path: Path
    success: bool
    error_message: Optional[str] = None


class EvidenceLogger:
    """
    Manages incident directory creation, annotated snapshot rendering,
    pre-event + post-event MP4 video clip generation, and atomic JSON writing.
    Uses a bounded background thread pool so video encoding never blocks the
    central frame processing loop.
    """

    MAX_PENDING_JOBS: int = 32

    def __init__(
        self,
        storage_config: StorageConfig,
        corridor: CorridorGeometry,
        amr_config: AmrConfig,
    ) -> None:
        self._storage_cfg = storage_config
        self._corridor = corridor
        self._amr_cfg = amr_config
        self._base_dir = Path(storage_config.evidence_base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

        self._recorded_track_ids: Set[int] = set()
        self._pending_jobs: Dict[str, PendingIncidentJob] = {}
        self._completed_records: List[CompletedIncidentRecord] = []
        self._futures: List[Future[CompletedIncidentRecord]] = []
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="EvidenceWriter")

    @property
    def pending_job_count(self) -> int:
        """Return number of pending or actively encoding evidence jobs."""
        with self._lock:
            active_futures = sum(1 for f in self._futures if not f.done())
            return len(self._pending_jobs) + active_futures

    def has_incident_for_track(self, track_id: int) -> bool:
        """Return True if an incident has already been initiated or recorded for track_id."""
        with self._lock:
            return track_id in self._recorded_track_ids

    def start_incident(
        self,
        track: TrackedObject,
        risk_result: RiskAssessmentResult,
        frame_id: int,
        current_frame: np.ndarray,
        pre_event_frames: List[np.ndarray],
        is_preexisting_debris: bool = False,
        post_event_frames_override: Optional[int] = None,
    ) -> Optional[str]:
        """
        Register a new confirmed landing incident, freeze pre-event frames,
        and begin collecting post-event frames.
        """
        with self._lock:
            if track.track_id in self._recorded_track_ids:
                logger.warning(
                    "Duplicate incident prevented for track_id=%d", track.track_id
                )
                return None

            if len(self._pending_jobs) >= self.MAX_PENDING_JOBS:
                logger.error(
                    "Evidence queue overflow (%d active jobs); dropping incident for track_id=%d",
                    len(self._pending_jobs),
                    track.track_id,
                )
                return None

            if not self._check_disk_space():
                logger.error(
                    "Insufficient disk space (< %d MB free); cannot record incident for track_id=%d",
                    self._storage_cfg.disk_min_free_mb,
                    track.track_id,
                )
                return None

            self._recorded_track_ids.add(track.track_id)

        now_utc = datetime.now(timezone.utc)
        ts_folder = now_utc.strftime("%Y%m%dT%H%M%SZ")
        iso_ts = now_utc.isoformat()
        unique_short = uuid.uuid4().hex[:8].upper()
        safe_class = re.sub(r"[^A-Za-z0-9_]", "_", track.class_name).upper() or "OBJECT"

        folder_name = f"FALL_{safe_class}_TRK{track.track_id}_{ts_folder}_{unique_short}"
        incident_dir = self._base_dir / folder_name

        if post_event_frames_override is not None:
            post_frames_needed = max(0, int(post_event_frames_override))
        else:
            post_frames_needed = max(
                1,
                int(round(self._storage_cfg.camera_fps * self._storage_cfg.post_event_duration_sec)),
            )

        job = PendingIncidentJob(
            event_id=folder_name,
            folder_name=folder_name,
            incident_dir=incident_dir,
            track_snapshot=track,
            risk_result=risk_result,
            frame_id=frame_id,
            iso_timestamp=iso_ts,
            landing_frame=current_frame.copy(),
            pre_event_frames=[f.copy() for f in pre_event_frames],
            post_event_frames=[],
            remaining_post_frames=post_frames_needed,
            is_preexisting_debris=is_preexisting_debris,
        )

        with self._lock:
            if post_frames_needed == 0:
                fut = self._executor.submit(self._finalize_incident_job, job)
                self._futures.append(fut)
            else:
                self._pending_jobs[job.event_id] = job

        return folder_name

    def ingest_post_event_frame(self, frame: np.ndarray) -> List[CompletedIncidentRecord]:
        """
        Append a copy of the latest frame to all pending post-event jobs.
        When a job collects all required post-event frames, dispatch it to the background writer.
        """
        ready_jobs: List[PendingIncidentJob] = []
        with self._lock:
            for event_id, job in list(self._pending_jobs.items()):
                job.post_event_frames.append(frame.copy())
                job.remaining_post_frames -= 1
                if job.remaining_post_frames <= 0:
                    ready_jobs.append(job)
                    del self._pending_jobs[event_id]

            for job in ready_jobs:
                fut = self._executor.submit(self._finalize_incident_job, job)
                self._futures.append(fut)

        return self.harvest_completed_records()

    def harvest_completed_records(self) -> List[CompletedIncidentRecord]:
        """Collect any background evidence jobs that have finished writing."""
        newly_completed: List[CompletedIncidentRecord] = []
        with self._lock:
            still_running: List[Future[CompletedIncidentRecord]] = []
            for fut in self._futures:
                if fut.done():
                    try:
                        rec = fut.result()
                        self._completed_records.append(rec)
                        newly_completed.append(rec)
                    except Exception as exc:
                        logger.error("Unhandled exception in evidence writer future: %s", exc)
                else:
                    still_running.append(fut)
            self._futures = still_running
        return newly_completed

    def flush_and_shutdown(self) -> List[CompletedIncidentRecord]:
        """
        Finalize any partially collected pending incidents immediately, wait for
        all background encoding tasks to complete, and shut down the executor.
        """
        with self._lock:
            remaining = list(self._pending_jobs.values())
            self._pending_jobs.clear()
            for job in remaining:
                fut = self._executor.submit(self._finalize_incident_job, job)
                self._futures.append(fut)
            futures_to_wait = list(self._futures)

        for fut in futures_to_wait:
            try:
                fut.result(timeout=30.0)
            except Exception as exc:
                logger.error("Error flushing evidence job during shutdown: %s", exc)

        self.harvest_completed_records()
        self._executor.shutdown(wait=True)
        return list(self._completed_records)

    def _check_disk_space(self) -> bool:
        """Verify that available free space on the target filesystem meets minimum requirements."""
        try:
            usage = shutil.disk_usage(self._base_dir)
            free_mb = usage.free / (1024 * 1024)
            return free_mb >= self._storage_cfg.disk_min_free_mb
        except OSError as exc:
            logger.error("Disk space check failed: %s", exc)
            return False

    def _finalize_incident_job(self, job: PendingIncidentJob) -> CompletedIncidentRecord:
        """Write snapshot.jpg, incident_clip.mp4, and metadata.json and verify all three files."""
        incident_dir = job.incident_dir
        snapshot_path = incident_dir / "snapshot.jpg"
        video_path = incident_dir / "incident_clip.mp4"
        metadata_path = incident_dir / "metadata.json"

        try:
            incident_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            # Guard against accidental collision
            unique_suffix = uuid.uuid4().hex[:6].upper()
            incident_dir = self._base_dir / f"{job.folder_name}_{unique_suffix}"
            incident_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = incident_dir / "snapshot.jpg"
            video_path = incident_dir / "incident_clip.mp4"
            metadata_path = incident_dir / "metadata.json"
        except OSError as exc:
            err = f"Failed to create incident directory {incident_dir}: {exc}"
            logger.error(err)
            return CompletedIncidentRecord(
                event_id=job.event_id,
                track_id=job.track_snapshot.track_id,
                incident_dir=incident_dir,
                snapshot_path=snapshot_path,
                video_path=video_path,
                metadata_path=metadata_path,
                success=False,
                error_message=err,
            )

        try:
            # 1. Render and save annotated snapshot.jpg
            annotated_snapshot = self.render_incident_snapshot(
                job.landing_frame,
                job.track_snapshot,
                job.risk_result,
                job.iso_timestamp,
            )
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self._storage_cfg.jpeg_quality)]
            ok_snap = cv2.imwrite(str(snapshot_path), annotated_snapshot, encode_params)
            if not ok_snap or not snapshot_path.exists() or snapshot_path.stat().st_size == 0:
                raise RuntimeError(f"Failed to write valid snapshot image at {snapshot_path}")

            # 2. Encode pre + post event frames into incident_clip.mp4
            all_frames = list(job.pre_event_frames)
            if not all_frames:
                all_frames.append(job.landing_frame)
            all_frames.extend(job.post_event_frames)

            self._write_video_clip(video_path, all_frames)
            if not video_path.exists() or video_path.stat().st_size == 0:
                raise RuntimeError(f"Failed to write valid video clip at {video_path}")

            # 3. Write metadata.json atomically
            metadata_dict = self._build_metadata_dict(job)
            self._write_json_atomically(metadata_path, metadata_dict)
            if not metadata_path.exists() or metadata_path.stat().st_size == 0:
                raise RuntimeError(f"Failed to write valid metadata JSON at {metadata_path}")

            logger.info(
                "Incident evidence verified and saved: %s (risk=%s)",
                incident_dir,
                job.risk_result.risk_level.value,
            )
            return CompletedIncidentRecord(
                event_id=job.event_id,
                track_id=job.track_snapshot.track_id,
                incident_dir=incident_dir,
                snapshot_path=snapshot_path,
                video_path=video_path,
                metadata_path=metadata_path,
                success=True,
                error_message=None,
            )
        except Exception as exc:
            err_msg = f"Evidence generation failed for {job.event_id}: {exc}"
            logger.error(err_msg)
            return CompletedIncidentRecord(
                event_id=job.event_id,
                track_id=job.track_snapshot.track_id,
                incident_dir=incident_dir,
                snapshot_path=snapshot_path,
                video_path=video_path,
                metadata_path=metadata_path,
                success=False,
                error_message=err_msg,
            )

    def render_incident_snapshot(
        self,
        frame: np.ndarray,
        track: TrackedObject,
        risk: RiskAssessmentResult,
        iso_timestamp: str,
    ) -> np.ndarray:
        """
        Render all required visual elements onto the incident snapshot:
        - Yellow centerline
        - Corridor polygon
        - Intersection area highlight
        - Segmentation contour / footprint
        - Resting bounding box
        - Object class, confidence, track ID
        - Risk classification and overlap percentage
        - Timestamp
        """
        canvas = frame.copy()
        overlay = canvas.copy()

        # Draw corridor polygon
        corr_pts = np.array(self._corridor.vertices_pixel, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [corr_pts], (200, 200, 50))
        cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0, canvas)
        cv2.polylines(canvas, [corr_pts], isClosed=True, color=(0, 255, 255), thickness=2)

        # Draw yellow centerline
        p1 = (int(round(self._corridor.p1_pixel[0])), int(round(self._corridor.p1_pixel[1])))
        p2 = (int(round(self._corridor.p2_pixel[0])), int(round(self._corridor.p2_pixel[1])))
        cv2.line(canvas, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw intersection polygon if present and in pixel mode
        if (
            risk.intersection_polygon is not None
            and not risk.intersection_polygon.is_empty
            and self._corridor.geometry_mode == "pixel"
        ):
            inter_coords = np.array(
                list(risk.intersection_polygon.exterior.coords), dtype=np.int32
            ).reshape((-1, 1, 2))
            inter_overlay = canvas.copy()
            cv2.fillPoly(inter_overlay, [inter_coords], (0, 0, 255))
            cv2.addWeighted(inter_overlay, 0.45, canvas, 0.55, 0, canvas)
            cv2.polylines(canvas, [inter_coords], isClosed=True, color=(0, 0, 255), thickness=2)

        # Determine risk color
        risk_color_map = {
            RiskLevel.HIGH: (0, 0, 255),
            RiskLevel.MEDIUM: (0, 0, 255),
            RiskLevel.LOW: (0, 200, 0),
            RiskLevel.UNCERTAIN: (160, 160, 160),
        }
        color = risk_color_map.get(risk.risk_level, (160, 160, 160))

        # Draw segmentation / footprint contour
        if track.footprint is not None and not track.footprint.is_empty:
            fp_pts = np.array(list(track.footprint.exterior.coords), dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [fp_pts], isClosed=True, color=color, thickness=2)

        # Draw resting bounding box
        x1, y1, x2, y2 = [int(round(v)) for v in track.bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw annotations
        header_text = (
            f"INCIDENT | ID:{track.track_id} {track.class_name} "
            f"Conf:{track.confidence:.2f}"
        )
        risk_text = (
            f"RISK: {risk.risk_level.value} | Overlap: {risk.overlap_percentage:.1f}% "
            f"| InterArea: {risk.intersection_area:.1f}"
        )
        ts_text = f"Timestamp: {iso_timestamp}"

        cv2.rectangle(canvas, (10, 10), (min(canvas.shape[1] - 10, 620), 95), (20, 20, 20), -1)
        cv2.putText(
            canvas, header_text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            canvas, risk_text, (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA
        )
        cv2.putText(
            canvas, ts_text, (18, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA
        )

        label_y = max(25, y1 - 10)
        cv2.putText(
            canvas,
            f"TRK {track.track_id} {track.class_name} [{risk.risk_level.value}]",
            (max(5, x1), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        return canvas

    def _write_video_clip(self, video_path: Path, frames: List[np.ndarray]) -> None:
        """Encode frames into an MP4 file using OpenCV VideoWriter with validation."""
        if not frames:
            raise ValueError("Cannot write incident video clip with zero frames.")

        first = frames[0]
        height, width = first.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid frame dimensions: {width}x{height}")

        fourcc = cv2.VideoWriter_fourcc(*self._storage_cfg.video_codec)
        fps = float(max(1, self._storage_cfg.camera_fps))

        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(
                f"OpenCV VideoWriter failed to initialize codec '{self._storage_cfg.video_codec}' "
                f"for path '{video_path}'."
            )

        try:
            for frm in frames:
                if frm is None or frm.size == 0:
                    continue
                if frm.shape[0] != height or frm.shape[1] != width:
                    frm = cv2.resize(frm, (width, height))
                writer.write(frm)
        finally:
            writer.release()

    def _build_metadata_dict(self, job: PendingIncidentJob) -> Dict[str, Any]:
        """Build the structured metadata dictionary conforming to the exact schema."""
        track = job.track_snapshot
        risk = job.risk_result
        return {
            "event_id": str(job.event_id),
            "timestamp": str(job.iso_timestamp),
            "frame_id": int(job.frame_id),
            "detected_class": str(track.class_name),
            "track_id": int(track.track_id),
            "confidence": round(float(track.confidence), 4),
            "flags": {
                "is_preexisting_debris": bool(job.is_preexisting_debris),
                "fallen_on_to_ground": True,
            },
            "corridor_parameters": {
                "yellow_line_p1": list(self._corridor.p1_pixel),
                "yellow_line_p2": list(self._corridor.p2_pixel),
                "geometry_mode": str(self._corridor.geometry_mode),
                "amr_base_half_width": float(self._amr_cfg.base_half_width),
                "amr_safety_margin": float(self._amr_cfg.safety_margin),
                "total_half_width_working": float(self._corridor.half_width_working),
            },
            "spatial_analysis": {
                "object_area": round(float(risk.object_area), 4),
                "intersection_area": round(float(risk.intersection_area), 4),
                "overlap_percentage": round(float(risk.overlap_percentage), 4),
                "touches_boundary": bool(risk.touches_boundary),
                "external_distance": round(float(risk.external_distance), 4),
                "geometry_valid": bool(risk.geometry_valid),
            },
            "risk_assessment": {
                "risk_level": str(risk.risk_level.value),
                "justification": str(risk.justification),
            },
            "evidence_files": {
                "snapshot": "snapshot.jpg",
                "video_clip": "incident_clip.mp4",
            },
        }

    @staticmethod
    def _write_json_atomically(target_path: Path, payload: Dict[str, Any]) -> None:
        """Serialize payload to JSON and atomically replace target_path."""
        temp_path = target_path.with_suffix(".json.tmp")
        serialized = json.dumps(payload, indent=2, ensure_ascii=False)
        try:
            with open(temp_path, "w", encoding="utf-8") as fh:
                fh.write(serialized)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temp_path, target_path)
        except Exception as exc:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise RuntimeError(f"Atomic JSON write failed for {target_path}: {exc}") from exc
