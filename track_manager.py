"""Selective ByteTrack integration and persistent falling-object track management."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
import math
import time
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from baseline_profiler import compute_bbox_iou, compute_euclidean_distance
from config_manager import LandingConfig, MotionConfig, TrackingConfig
from detector import Detection
from motion_gate import EvaluatedDetection, LifecycleState

logger = logging.getLogger(__name__)


@dataclass
class ByteTrackInternalTracklet:
    """Internal tracklet state for the built-in ByteTrack two-stage association engine."""

    internal_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    velocity: np.ndarray  # [vx1, vy1, vx2, vy2] per frame
    score: float
    class_id: int
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    is_activated: bool = False

    def predict(self) -> np.ndarray:
        """Linear constant-velocity state prediction for bounding box coordinates."""
        if self.time_since_update > 0:
            self.velocity *= 0.85
        self.bbox = self.bbox + self.velocity
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def update(self, new_bbox: np.ndarray, score: float, class_id: int) -> None:
        """Update state estimate with a matched detection bounding box."""
        alpha = 0.6
        measured_vel = new_bbox - self.bbox
        self.velocity = alpha * measured_vel + (1.0 - alpha) * self.velocity
        self.bbox = new_bbox.copy()
        self.score = score
        self.class_id = class_id
        self.hits += 1
        self.time_since_update = 0
        self.is_activated = True


class NativeByteTracker:
    """
    Standard two-stage ByteTrack association engine (Zhang et al., ECCV 2022).

    Used when supervision.ByteTrack is unavailable or as the deterministic fallback,
    implementing identical high-threshold primary association and low-threshold
    secondary association for occluded/decelerating objects.
    """

    def __init__(
        self,
        high_threshold: float,
        low_threshold: float,
        match_threshold: float,
        track_buffer: int,
    ) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_threshold = match_threshold
        self.track_buffer = track_buffer
        self._tracklets: Dict[int, ByteTrackInternalTracklet] = {}
        self._next_id: int = 1

    def update(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Execute two-stage ByteTrack update.

        Returns list of (detection_index, tracker_id) for all matched/instantiated detections.
        """
        # 1. Predict all existing tracklets
        active_ids = list(self._tracklets.keys())
        for tid in active_ids:
            self._tracklets[tid].predict()

        if len(bboxes) == 0:
            self._prune_expired()
            return []

        high_indices = [i for i, s in enumerate(scores) if s >= self.high_threshold]
        low_indices = [
            i for i, s in enumerate(scores) if self.low_threshold <= s < self.high_threshold
        ]

        matched_pairs: List[Tuple[int, int]] = []
        unmatched_track_ids = list(active_ids)

        # Stage 1: Associate high-confidence detections with all active/lost tracklets
        if high_indices and unmatched_track_ids:
            matches_1, unmatched_tracks_1, unmatched_dets_1 = self._associate(
                unmatched_track_ids, high_indices, bboxes, self.match_threshold
            )
            for tid, det_idx in matches_1:
                self._tracklets[tid].update(bboxes[det_idx], float(scores[det_idx]), int(class_ids[det_idx]))
                matched_pairs.append((det_idx, tid))
            unmatched_track_ids = unmatched_tracks_1
            rem_high_dets = unmatched_dets_1
        else:
            rem_high_dets = list(high_indices)

        # Stage 2: Associate low-confidence detections with remaining unmatched tracklets
        if low_indices and unmatched_track_ids:
            matches_2, unmatched_tracks_2, _ = self._associate(
                unmatched_track_ids, low_indices, bboxes, max(0.15, self.match_threshold * 0.5)
            )
            for tid, det_idx in matches_2:
                self._tracklets[tid].update(bboxes[det_idx], float(scores[det_idx]), int(class_ids[det_idx]))
                matched_pairs.append((det_idx, tid))
            unmatched_track_ids = unmatched_tracks_2

        # Stage 3: Instantiate new tracklets for unmatched high-confidence detections
        for det_idx in rem_high_dets:
            new_id = self._next_id
            self._next_id += 1
            self._tracklets[new_id] = ByteTrackInternalTracklet(
                internal_id=new_id,
                bbox=bboxes[det_idx].copy(),
                velocity=np.zeros(4, dtype=np.float64),
                score=float(scores[det_idx]),
                class_id=int(class_ids[det_idx]),
                age=1,
                hits=1,
                time_since_update=0,
                is_activated=True,
            )
            matched_pairs.append((det_idx, new_id))

        self._prune_expired()
        return matched_pairs

    def _associate(
        self,
        track_ids: List[int],
        det_indices: List[int],
        bboxes: np.ndarray,
        iou_Gate: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Solve linear assignment between predicted tracklet boxes and detection boxes."""
        cost_matrix = np.ones((len(track_ids), len(det_indices)), dtype=np.float64)
        for r, tid in enumerate(track_ids):
            t_box = tuple(float(v) for v in self._tracklets[tid].bbox)
            for c, didx in enumerate(det_indices):
                d_box = tuple(float(v) for v in bboxes[didx])
                iou = compute_bbox_iou(t_box, d_box)
                tcx = (t_box[0] + t_box[2]) * 0.5
                tcy = (t_box[1] + t_box[3]) * 0.5
                dcx = (d_box[0] + d_box[2]) * 0.5
                dcy = (d_box[1] + d_box[3]) * 0.5
                dist = math.hypot(tcx - dcx, tcy - dcy)
                dist_similarity = max(0.0, 1.0 - dist / 150.0)
                combined_sim = max(iou, iou * 0.6 + dist_similarity * 0.4)
                cost_matrix[r, c] = 1.0 - combined_sim

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches: List[Tuple[int, int]] = []
        matched_rows: Set[int] = set()
        matched_cols: Set[int] = set()

        min_sim = min(0.20, iou_Gate * 0.35)
        for r, c in zip(row_ind, col_ind):
            sim = 1.0 - cost_matrix[r, c]
            if sim >= min_sim:
                matches.append((track_ids[r], det_indices[c]))
                matched_rows.add(r)
                matched_cols.add(c)

        unmatched_tracks = [track_ids[r] for r in range(len(track_ids)) if r not in matched_rows]
        unmatched_dets = [det_indices[c] for c in range(len(det_indices)) if c not in matched_cols]
        return matches, unmatched_tracks, unmatched_dets

    def _prune_expired(self) -> None:
        """Remove tracklets that have exceeded the configured track buffer."""
        for tid in list(self._tracklets.keys()):
            if self._tracklets[tid].time_since_update > self.track_buffer:
                del self._tracklets[tid]


@dataclass
class TrackedObject:
    """Comprehensive persistent state of a tracked falling object."""

    track_id: int
    external_tracker_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    footprint: Polygon
    has_segmentation_mask: bool
    mask_contour: Optional[List[Tuple[float, float]]]
    position_history: Deque[Tuple[float, float, float]] = field(default_factory=deque)  # (x, y, timestamp)
    velocity_history: Deque[Tuple[float, float]] = field(default_factory=deque)  # (vx, vy)
    acceleration_history: Deque[Tuple[float, float]] = field(default_factory=deque)  # (ax, ay)
    current_vx: float = 0.0
    current_vy: float = 0.0
    current_ax: float = 0.0
    current_ay: float = 0.0
    peak_downward_vy: float = 0.0
    track_age: int = 1
    total_observations: int = 1
    missed_frames: int = 0
    lifecycle_state: LifecycleState = LifecycleState.FALLING
    landing_counter: int = 0
    has_qualified_as_falling: bool = True
    incident_generated: bool = False
    resting_anchor_centroid: Optional[Tuple[float, float]] = None
    first_seen_timestamp: float = 0.0
    last_seen_timestamp: float = 0.0
    landed_timestamp: Optional[float] = None
    landed_frame_id: Optional[int] = None
    risk_level: Optional[str] = None
    overlap_percentage: float = 0.0
    corridor_contact: bool = False


@dataclass(frozen=True)
class TrackingFrameOutput:
    """Structured output returned by TrackManager to main.py."""

    active_tracks: List[TrackedObject]
    newly_landed_tracks: List[TrackedObject]
    expired_track_ids: List[int]
    tracking_latency_ms: float


class TrackManager:
    """
    Manages selective ByteTrack execution, persistent falling-object state,
    occlusion survival, fragmentation recovery, and landing verification.
    """

    def __init__(
        self,
        tracking_config: TrackingConfig,
        motion_config: MotionConfig,
        landing_config: LandingConfig,
        camera_fps: int = 30,
    ) -> None:
        self._tracking_cfg = tracking_config
        self._motion_cfg = motion_config
        self._landing_cfg = landing_config
        self._camera_fps = max(1, camera_fps)

        self._history_maxlen = max(
            30,
            tracking_config.track_buffer_frames + landing_config.rest_frames + 20,
        )

        self._sv_byte_track: Optional[Any] = None
        self._sv_module: Optional[Any] = None
        self._init_supervision_bytetrack()

        self._native_byte_track = NativeByteTracker(
            high_threshold=tracking_config.byte_track_high_threshold,
            low_threshold=tracking_config.byte_track_low_threshold,
            match_threshold=tracking_config.byte_track_match_threshold,
            track_buffer=tracking_config.track_buffer_frames,
        )

        self._tracks: Dict[int, TrackedObject] = {}
        self._ext_to_internal_id: Dict[int, int] = {}
        self._retired_internal_ids: Set[int] = set()
        self._next_internal_id: int = 1

    def _init_supervision_bytetrack(self) -> None:
        """Initialize supervision.ByteTrack if the supervision package is installed."""
        try:
            import supervision as sv

            self._sv_module = sv
            self._sv_byte_track = sv.ByteTrack(
                track_activation_threshold=self._tracking_cfg.byte_track_high_threshold,
                lost_track_buffer=self._tracking_cfg.track_buffer_frames,
                minimum_matching_threshold=self._tracking_cfg.byte_track_match_threshold,
                frame_rate=self._camera_fps,
            )
            logger.info("Initialized supervision.ByteTrack backend.")
        except Exception as exc:
            self._sv_byte_track = None
            self._sv_module = None
            logger.info(
                "Using built-in NativeByteTracker backend (supervision not active: %s).",
                exc,
            )

    def get_active_track_bboxes(self) -> List[Tuple[float, float, float, float]]:
        """Return bounding boxes of all currently active tracks for selective association."""
        return [
            t.bbox
            for t in self._tracks.values()
            if t.lifecycle_state
            not in (LifecycleState.EXPIRED, LifecycleState.TRACK_LOST, LifecycleState.REJECTED)
        ]

    def get_active_tracks(self) -> List[TrackedObject]:
        """Return all active tracks."""
        return [
            t
            for t in self._tracks.values()
            if t.lifecycle_state
            not in (LifecycleState.EXPIRED, LifecycleState.TRACK_LOST, LifecycleState.REJECTED)
        ]

    def deduplicate_detections(
        self, evaluated_detections: List[EvaluatedDetection]
    ) -> List[EvaluatedDetection]:
        """Suppress duplicate overlapping detections of the same object in a single frame."""
        if len(evaluated_detections) <= 1:
            return list(evaluated_detections)

        sorted_dets = sorted(
            evaluated_detections,
            key=lambda ed: ed.detection.confidence,
            reverse=True,
        )
        kept: List[EvaluatedDetection] = []
        for candidate in sorted_dets:
            is_duplicate = False
            for existing in kept:
                if candidate.detection.class_id == existing.detection.class_id:
                    iou = compute_bbox_iou(candidate.detection.bbox, existing.detection.bbox)
                    if iou >= 0.85:
                        is_duplicate = True
                        break
            if not is_duplicate:
                kept.append(candidate)
        return kept

    def update(
        self,
        qualifying_detections: List[EvaluatedDetection],
        all_non_debris_detections: List[EvaluatedDetection],
        frame_id: int,
        timestamp: float,
    ) -> TrackingFrameOutput:
        """
        Update ByteTrack selectively with qualifying falling candidates and detections
        required to maintain already-activated tracks (e.g., during deceleration/landing).
        """
        start_time = time.perf_counter()

        # Select detections: qualifying falling candidates + detections matching active tracks
        selected_map: Dict[int, EvaluatedDetection] = {
            ed.detection.detection_id: ed for ed in qualifying_detections
        }
        for ed in all_non_debris_detections:
            if ed.detection.detection_id in selected_map:
                continue
            if self._matches_any_active_track(ed.detection):
                selected_map[ed.detection.detection_id] = ed

        selected_eval_dets = self.deduplicate_detections(list(selected_map.values()))

        # Run ByteTrack association
        det_to_ext_id = self._run_bytetrack(selected_eval_dets)

        matched_internal_ids: Set[int] = set()

        for det_idx, ext_id in det_to_ext_id:
            eval_det = selected_eval_dets[det_idx]
            det = eval_det.detection

            internal_id = self._resolve_internal_track_id(ext_id, det, matched_internal_ids)

            if internal_id is not None and internal_id in self._tracks:
                track = self._tracks[internal_id]
                self._update_existing_track(track, eval_det, timestamp)
                matched_internal_ids.add(internal_id)
            else:
                # Only instantiate a brand-new track if it qualified as a descending candidate
                # and we have capacity
                if not eval_det.qualifies_for_bytetrack:
                    continue
                if len(self._tracks) >= self._tracking_cfg.maximum_active_tracks:
                    logger.warning(
                        "Maximum active tracks (%d) reached; dropping new track candidate.",
                        self._tracking_cfg.maximum_active_tracks,
                    )
                    continue

                new_internal_id = self._allocate_internal_id()
                self._ext_to_internal_id[ext_id] = new_internal_id
                new_track = self._create_new_track(
                    new_internal_id, ext_id, eval_det, timestamp
                )
                self._tracks[new_internal_id] = new_track
                matched_internal_ids.add(new_internal_id)

        # Fallback association for active tracks that ByteTrack missed (fragmentation recovery)
        unmatched_eval_dets = [
            ed
            for idx, ed in enumerate(selected_eval_dets)
            if idx not in {d_idx for d_idx, _ in det_to_ext_id}
        ]
        if unmatched_eval_dets:
            self._recover_fragmented_tracks(
                unmatched_eval_dets, matched_internal_ids, timestamp
            )

        # Age unmatched tracks (temporary occlusion handling)
        expired_ids: List[int] = []
        for tid in list(self._tracks.keys()):
            if tid not in matched_internal_ids:
                track = self._tracks[tid]
                track.track_age += 1
                track.missed_frames += 1

                # Do NOT increment landing_counter during missed detections!
                if track.missed_frames > self._landing_cfg.maximum_detection_gap_frames:
                    if track.lifecycle_state == LifecycleState.POSSIBLE_LANDING:
                        track.landing_counter = 0
                        track.resting_anchor_centroid = None
                        track.lifecycle_state = LifecycleState.FALLING

                if track.missed_frames > self._tracking_cfg.maximum_missed_frames:
                    track.lifecycle_state = LifecycleState.EXPIRED
                    expired_ids.append(tid)
                    self._retired_internal_ids.add(tid)
                    del self._tracks[tid]

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return TrackingFrameOutput(
            active_tracks=list(self._tracks.values()),
            newly_landed_tracks=[],
            expired_track_ids=expired_ids,
            tracking_latency_ms=latency_ms,
        )

    def verify_landing(self, frame_id: int, timestamp: float) -> List[TrackedObject]:
        """
        Evaluate landing criteria across all active tracks and return tracks that
        newly transitioned to LANDED on this frame.

        Landing condition:
            abs(vy) <= landing.rest_max_velocity
            for landing.rest_frames consecutive valid observations.
        Additional safeguards:
            - Track must have previously qualified as falling.
            - Track age >= landing.minimum_track_age_frames and observations >= minimum_track_observations.
            - Position must remain stable within landing.position_stability_threshold_px.
            - Temporary detection loss (missed_frames > 0) does NOT increment the resting counter.
            - Movement or bounce resets the resting counter.
        """
        newly_landed: List[TrackedObject] = []

        for track in self._tracks.values():
            if track.lifecycle_state in (
                LifecycleState.LANDED,
                LifecycleState.RISK_ASSESSED,
                LifecycleState.INCIDENT_RECORDED,
                LifecycleState.EXPIRED,
                LifecycleState.TRACK_LOST,
                LifecycleState.REJECTED,
            ):
                continue

            # Temporary detection loss must never increment resting counter or confirm landing
            if track.missed_frames > 0:
                continue

            if not track.has_qualified_as_falling:
                continue

            is_slow_vertical = abs(track.current_vy) <= self._landing_cfg.rest_max_velocity
            is_slow_horizontal = abs(track.current_vx) <= self._landing_cfg.rest_max_velocity * 1.5

            if is_slow_vertical and is_slow_horizontal:
                if track.resting_anchor_centroid is None:
                    track.resting_anchor_centroid = track.centroid
                    track.landing_counter = 1
                    track.lifecycle_state = LifecycleState.POSSIBLE_LANDING
                else:
                    drift = compute_euclidean_distance(
                        track.centroid, track.resting_anchor_centroid
                    )
                    if drift <= self._landing_cfg.position_stability_threshold_px:
                        track.landing_counter += 1
                        track.lifecycle_state = LifecycleState.POSSIBLE_LANDING
                    else:
                        # Object rolled or shifted beyond stability threshold -> reset anchor
                        track.resting_anchor_centroid = track.centroid
                        track.landing_counter = 1
                        track.lifecycle_state = LifecycleState.POSSIBLE_LANDING

                if (
                    track.landing_counter >= self._landing_cfg.rest_frames
                    and track.track_age >= self._landing_cfg.minimum_track_age_frames
                    and track.total_observations >= self._tracking_cfg.minimum_track_observations
                ):
                    track.lifecycle_state = LifecycleState.LANDED
                    track.landed_timestamp = timestamp
                    track.landed_frame_id = frame_id
                    newly_landed.append(track)
            else:
                # Object is still falling or bounced/resumed motion -> reset resting counter
                track.landing_counter = 0
                track.resting_anchor_centroid = None
                track.lifecycle_state = LifecycleState.FALLING

        return newly_landed

    def _run_bytetrack(
        self, selected_dets: List[EvaluatedDetection]
    ) -> List[Tuple[int, int]]:
        """Run ByteTrack on selected detections and return (det_idx, external_tracker_id) pairs."""
        if not selected_dets:
            self._native_byte_track.update(
                np.empty((0, 4), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.int64),
            )
            return []

        bboxes = np.array(
            [ed.detection.bbox for ed in selected_dets], dtype=np.float64
        )
        scores = np.array(
            [ed.detection.confidence for ed in selected_dets], dtype=np.float64
        )
        class_ids = np.array(
            [ed.detection.class_id for ed in selected_dets], dtype=np.int64
        )

        if self._sv_byte_track is not None and self._sv_module is not None:
            try:
                sv_dets = self._sv_module.Detections(
                    xyxy=bboxes,
                    confidence=scores,
                    class_id=class_ids,
                )
                tracked_dets = self._sv_byte_track.update_with_detections(sv_dets)
                pairs: List[Tuple[int, int]] = []
                if tracked_dets.tracker_id is not None:
                    for t_idx in range(len(tracked_dets.xyxy)):
                        t_box = tuple(float(v) for v in tracked_dets.xyxy[t_idx])
                        t_id = int(tracked_dets.tracker_id[t_idx])
                        best_d_idx = -1
                        best_iou = 0.0
                        for d_idx, ed in enumerate(selected_dets):
                            iou = compute_bbox_iou(t_box, ed.detection.bbox)
                            if iou > best_iou:
                                best_iou = iou
                                best_d_idx = d_idx
                        if best_d_idx >= 0 and best_iou >= 0.5:
                            pairs.append((best_d_idx, t_id))
                if pairs:
                    return pairs
            except Exception as exc:
                logger.warning("supervision.ByteTrack fallback to NativeByteTracker: %s", exc)

        return self._native_byte_track.update(bboxes, scores, class_ids)

    def _resolve_internal_track_id(
        self,
        ext_id: int,
        det: Detection,
        already_matched_internal: Set[int],
    ) -> Optional[int]:
        """
        Resolve external ByteTrack ID to a validated internal track ID, preventing
        corrupted associations when an external tracker reuses an ID for a distant object.
        """
        if ext_id in self._ext_to_internal_id:
            candidate_internal = self._ext_to_internal_id[ext_id]
            if (
                candidate_internal not in self._retired_internal_ids
                and candidate_internal in self._tracks
                and candidate_internal not in already_matched_internal
            ):
                track = self._tracks[candidate_internal]
                iou = compute_bbox_iou(det.bbox, track.bbox)
                dist = compute_euclidean_distance(det.centroid, track.centroid)
                max_allowed_dist = self._tracking_cfg.track_match_distance_px * max(
                    1, track.missed_frames + 1
                )
                if track.class_id == det.class_id and (
                    iou >= self._tracking_cfg.track_match_iou * 0.5 or dist <= max_allowed_dist
                ):
                    return candidate_internal
                # External ID was reused for an incompatible spatial location -> decouple
                del self._ext_to_internal_id[ext_id]

        # Attempt spatial match against active unmatched tracks (handles tracker ID switches)
        best_internal_id: Optional[int] = None
        best_score = 0.0
        for tid, track in self._tracks.items():
            if tid in already_matched_internal:
                continue
            if track.class_id != det.class_id:
                continue
            iou = compute_bbox_iou(det.bbox, track.bbox)
            dist = compute_euclidean_distance(det.centroid, track.centroid)
            if iou >= self._tracking_cfg.track_match_iou or dist <= self._tracking_cfg.track_match_distance_px:
                score = iou + max(0.0, 1.0 - dist / self._tracking_cfg.track_match_distance_px) * 0.5
                if score > best_score:
                    best_score = score
                    best_internal_id = tid

        if best_internal_id is not None:
            self._ext_to_internal_id[ext_id] = best_internal_id
            return best_internal_id

        return None

    def _allocate_internal_id(self) -> int:
        """Allocate a guaranteed unique monotonic internal track ID."""
        while self._next_internal_id in self._retired_internal_ids:
            self._next_internal_id += 1
        allocated = self._next_internal_id
        self._next_internal_id += 1
        return allocated

    def _create_new_track(
        self,
        internal_id: int,
        ext_id: int,
        eval_det: EvaluatedDetection,
        timestamp: float,
    ) -> TrackedObject:
        """Create a new TrackedObject initialized from a qualifying EvaluatedDetection."""
        det = eval_det.detection
        pos_hist: Deque[Tuple[float, float, float]] = deque(maxlen=self._history_maxlen)
        pos_hist.append((det.centroid[0], det.centroid[1], timestamp))

        vel_hist: Deque[Tuple[float, float]] = deque(maxlen=self._history_maxlen)
        vel_hist.append((eval_det.smoothed_vx, eval_det.smoothed_vy))

        acc_hist: Deque[Tuple[float, float]] = deque(maxlen=self._history_maxlen)
        acc_hist.append((0.0, 0.0))

        return TrackedObject(
            track_id=internal_id,
            external_tracker_id=ext_id,
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
            bbox=det.bbox,
            centroid=det.centroid,
            footprint=det.footprint,
            has_segmentation_mask=det.has_segmentation_mask,
            mask_contour=det.mask_contour,
            position_history=pos_hist,
            velocity_history=vel_hist,
            acceleration_history=acc_hist,
            current_vx=eval_det.smoothed_vx,
            current_vy=eval_det.smoothed_vy,
            current_ax=0.0,
            current_ay=0.0,
            peak_downward_vy=max(0.0, eval_det.smoothed_vy),
            track_age=1,
            total_observations=1,
            missed_frames=0,
            lifecycle_state=LifecycleState.FALLING,
            landing_counter=0,
            has_qualified_as_falling=True,
            incident_generated=False,
            resting_anchor_centroid=None,
            first_seen_timestamp=timestamp,
            last_seen_timestamp=timestamp,
        )

    def _update_existing_track(
        self,
        track: TrackedObject,
        eval_det: EvaluatedDetection,
        timestamp: float,
    ) -> None:
        """Update kinematics, footprint, and history for a matched active track."""
        det = eval_det.detection
        prev_x, prev_y, prev_t = track.position_history[-1]
        dt = timestamp - prev_t

        if dt > 1e-6:
            inst_vx = (det.centroid[0] - prev_x) / dt
            inst_vy = (det.centroid[1] - prev_y) / dt
        else:
            inst_vx = track.current_vx
            inst_vy = track.current_vy

        prev_vx, prev_vy = track.current_vx, track.current_vy
        track.velocity_history.append((inst_vx, inst_vy))

        window = min(len(track.velocity_history), self._motion_cfg.velocity_smoothing_window)
        recent_vels = list(track.velocity_history)[-window:]
        smoothed_vx = float(sum(v[0] for v in recent_vels) / window)
        smoothed_vy = float(sum(v[1] for v in recent_vels) / window)

        if dt > 1e-6:
            ax = (smoothed_vx - prev_vx) / dt
            ay = (smoothed_vy - prev_vy) / dt
        else:
            ax, ay = track.current_ax, track.current_ay

        track.current_vx = smoothed_vx
        track.current_vy = smoothed_vy
        track.current_ax = ax
        track.current_ay = ay
        track.peak_downward_vy = max(track.peak_downward_vy, smoothed_vy)

        track.position_history.append((det.centroid[0], det.centroid[1], timestamp))
        track.acceleration_history.append((ax, ay))

        track.confidence = det.confidence
        track.bbox = det.bbox
        track.centroid = det.centroid
        track.footprint = det.footprint
        track.has_segmentation_mask = det.has_segmentation_mask
        track.mask_contour = det.mask_contour
        track.track_age += 1
        track.total_observations += 1
        track.missed_frames = 0
        track.last_seen_timestamp = timestamp

        if eval_det.qualifies_for_bytetrack or smoothed_vy >= self._motion_cfg.descend_min_vy:
            track.has_qualified_as_falling = True

        # If a previously landed object starts falling or moving fast again, allow state update
        if track.lifecycle_state in (
            LifecycleState.LANDED,
            LifecycleState.RISK_ASSESSED,
            LifecycleState.INCIDENT_RECORDED,
        ):
            if abs(smoothed_vy) > self._landing_cfg.rest_max_velocity * 3.0:
                track.landing_counter = 0
                track.resting_anchor_centroid = None

    def _matches_any_active_track(self, det: Detection) -> bool:
        """Check whether a detection is spatially compatible with any active track."""
        for track in self._tracks.values():
            if track.class_id != det.class_id:
                continue
            iou = compute_bbox_iou(det.bbox, track.bbox)
            dist = compute_euclidean_distance(det.centroid, track.centroid)
            if iou >= self._tracking_cfg. byte_track_low_threshold or dist <= self._tracking_cfg.track_match_distance_px:
                return True
        return False

    def _recover_fragmented_tracks(
        self,
        unmatched_dets: List[EvaluatedDetection],
        matched_internal_ids: Set[int],
        timestamp: float,
    ) -> None:
        """Recover tracks that briefly fragmented due to occlusion or rapid deceleration."""
        for eval_det in unmatched_dets:
            det = eval_det.detection
            best_tid: Optional[int] = None
            best_dist = float("inf")

            for tid, track in self._tracks.items():
                if tid in matched_internal_ids:
                    continue
                if track.class_id != det.class_id:
                    continue
                iou = compute_bbox_iou(det.bbox, track.bbox)
                dist = compute_euclidean_distance(det.centroid, track.centroid)
                allowed_dist = self._tracking_cfg.track_match_distance_px * max(
                    1.0, 1.0 + 0.25 * track.missed_frames
                )
                if iou >= self._tracking_cfg.track_match_iou or dist <= allowed_dist:
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = tid

            if best_tid is not None:
                self._update_existing_track(self._tracks[best_tid], eval_det, timestamp)
                matched_internal_ids.add(best_tid)
