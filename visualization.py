"""OpenCV live visualization and keyboard interaction helper module."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from baseline_profiler import BaselineEntry
from config_manager import CorridorGeometry, DisplayConfig
from detector import Detection
from motion_gate import EvaluatedDetection, LifecycleState
from risk_evaluator import RiskAssessmentResult, RiskLevel
from track_manager import TrackedObject

logger = logging.getLogger(__name__)


# Exact BGR Color Constants per Specification
COLOR_YELLOW: Tuple[int, int, int] = (0, 255, 255)      # Corridor centerline
COLOR_ORANGE: Tuple[int, int, int] = (0, 165, 255)      # Falling candidates
COLOR_RED: Tuple[int, int, int] = (0, 0, 255)           # Confirmed landed objects & HIGH risk
COLOR_GREEN: Tuple[int, int, int] = (0, 200, 0)         # LOW risk
COLOR_GREY: Tuple[int, int, int] = (160, 160, 160)      # UNCERTAIN and baseline debris
COLOR_CYAN: Tuple[int, int, int] = (255, 255, 0)        # General object detections
COLOR_WHITE: Tuple[int, int, int] = (255, 255, 255)
COLOR_DARK_BG: Tuple[int, int, int] = (25, 25, 25)


@dataclass(frozen=True)
class VisualizationCommand:
    """User keyboard command captured during a single frame render step."""

    quit_requested: bool = False
    toggle_pause: bool = False
    save_screenshot: bool = False


class Visualizer:
    """
    Renders live telemetry, AMR corridor geometry, baseline debris, falling tracks,
    and risk classifications using OpenCV without owning an independent loop.
    """

    def __init__(self, config: DisplayConfig, corridor: CorridorGeometry) -> None:
        self._config = config
        self._corridor = corridor
        self._window_initialized = False

    def render(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        active_tracks: List[TrackedObject],
        baseline_debris: List[BaselineEntry],
        evaluated_detections: List[EvaluatedDetection],
        risk_by_track_id: Dict[int, RiskAssessmentResult],
        fps: float,
        incident_count: int,
        baseline_complete: bool,
    ) -> np.ndarray:
        """Draw all configured overlays onto a copy of frame and return the rendered BGR image."""
        canvas = frame.copy()

        # 1. Draw AMR corridor polygon and yellow centerline
        self._draw_corridor(canvas)

        # 2. Draw baseline stationary debris (Grey)
        if self._config.show_baseline_debris:
            for debris in baseline_debris:
                self._draw_baseline_debris(canvas, debris)

        # 3. Draw all current-frame detections, including objects not yet falling.
        for detection in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in detection.bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_CYAN, 1)
            cv2.putText(
                canvas,
                f"{detection.class_name} {detection.confidence:.2f}",
                (max(5, x1), max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_CYAN,
                1,
                cv2.LINE_AA,
            )

        # 4. Draw pre-association motion candidates not yet in active tracks
        tracked_bboxes = [t.bbox for t in active_tracks]
        for eval_det in evaluated_detections:
            if eval_det.state in (
                LifecycleState.MOTION_CANDIDATE,
                LifecycleState.DESCENDING_CANDIDATE,
            ):
                det = eval_det.detection
                if not any(
                    abs(det.bbox[0] - tb[0]) < 5.0 and abs(det.bbox[1] - tb[1]) < 5.0
                    for tb in tracked_bboxes
                ):
                    x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_ORANGE, 1)

        # 5. Draw active tracks (falling, possible landing, landed, risk-assessed)
        for track in active_tracks:
            risk = risk_by_track_id.get(track.track_id)
            self._draw_track(canvas, track, risk)

        # 6. Draw HUD telemetry panel
        self._draw_hud(
            canvas,
            fps=fps,
            active_track_count=len(active_tracks),
            incident_count=incident_count,
            baseline_complete=baseline_complete,
        )

        return canvas

    def show_and_poll_key(self, rendered_frame: np.ndarray, delay_ms: int = 1) -> VisualizationCommand:
        """
        If display is enabled, show the rendered frame in the configured OpenCV window
        and poll a single key event ('q' quit, 'p' pause/resume, 's' screenshot).
        """
        if not self._config.enabled:
            return VisualizationCommand()

        try:
            if not self._window_initialized:
                cv2.namedWindow(self._config.window_name, cv2.WINDOW_NORMAL)
                self._window_initialized = True

            cv2.imshow(self._config.window_name, rendered_frame)
            key = cv2.waitKey(max(1, delay_ms)) & 0xFF
        except cv2.error as exc:
            logger.warning(
                "OpenCV GUI display unavailable in current environment (%s); disabling display window.",
                exc,
            )
            self._window_initialized = False
            return VisualizationCommand()

        if key == ord("q") or key == 27:
            return VisualizationCommand(quit_requested=True)
        if key == ord("p"):
            return VisualizationCommand(toggle_pause=True)
        if key == ord("s"):
            return VisualizationCommand(save_screenshot=True)
        return VisualizationCommand()

    @staticmethod
    def save_diagnostic_screenshot(frame: np.ndarray, output_dir: str | Path, frame_id: int) -> Path:
        """Save a diagnostic screenshot to disk and return the saved file path."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"diagnostic_frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        return out_path

    def close(self) -> None:
        """Destroy the OpenCV visualization window if initialized."""
        if self._window_initialized:
            try:
                cv2.destroyWindow(self._config.window_name)
            except cv2.error:
                pass
            self._window_initialized = False

    def _draw_corridor(self, canvas: np.ndarray) -> None:
        """Draw the AMR corridor boundary and yellow centerline."""
        overlay = canvas.copy()
        pts = np.array(self._corridor.vertices_pixel, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], (180, 180, 40))
        cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
        cv2.polylines(canvas, [pts], isClosed=True, color=COLOR_YELLOW, thickness=2)

        p1 = (int(round(self._corridor.p1_pixel[0])), int(round(self._corridor.p1_pixel[1])))
        p2 = (int(round(self._corridor.p2_pixel[0])), int(round(self._corridor.p2_pixel[1])))
        cv2.line(canvas, p1, p2, COLOR_YELLOW, 2, cv2.LINE_AA)

    def _draw_baseline_debris(self, canvas: np.ndarray, debris: BaselineEntry) -> None:
        """Render pre-existing stationary debris in grey."""
        x1, y1, x2, y2 = [int(round(v)) for v in debris.bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_GREY, 1)
        if debris.footprint is not None and not debris.footprint.is_empty:
            fp_pts = np.array(list(debris.footprint.exterior.coords), dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [fp_pts], isClosed=True, color=COLOR_GREY, thickness=1)
        cv2.putText(
            canvas,
            f"{debris.label} ({debris.class_name})",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_GREY,
            1,
            cv2.LINE_AA,
        )

    def _draw_track(
        self,
        canvas: np.ndarray,
        track: TrackedObject,
        risk: Optional[RiskAssessmentResult],
    ) -> None:
        """Render an active track's bounding box, contour, trajectory, velocity arrow, and risk state."""
        is_landed = track.lifecycle_state in (
            LifecycleState.LANDED,
            LifecycleState.RISK_ASSESSED,
            LifecycleState.INCIDENT_RECORDED,
        )

        # Falling candidates are orange; assessed tracks use their risk color.
        risk_color = COLOR_RED if is_landed else COLOR_ORANGE
        if risk is not None:
            if risk.risk_level == RiskLevel.HIGH:
                risk_color = COLOR_RED
            elif risk.risk_level == RiskLevel.MEDIUM:
                risk_color = COLOR_ORANGE
            elif risk.risk_level == RiskLevel.LOW:
                risk_color = COLOR_GREEN
            else:
                risk_color = COLOR_GREY
        elif track.lifecycle_state == LifecycleState.LANDING_UNCERTAIN:
            risk_color = COLOR_GREY
        box_color = risk_color

        x1, y1, x2, y2 = [int(round(v)) for v in track.bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

        # Draw segmentation or footprint contour
        if track.footprint is not None and not track.footprint.is_empty:
            fp_pts = np.array(list(track.footprint.exterior.coords), dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [fp_pts], isClosed=True, color=risk_color, thickness=2)

        # Draw historical trajectory
        if self._config.show_trajectories and len(track.position_history) >= 2:
            traj_pts = np.array(
                [(int(round(p[0])), int(round(p[1]))) for p in track.position_history],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            cv2.polylines(canvas, [traj_pts], isClosed=False, color=box_color, thickness=2)

        # Draw vertical velocity motion arrow
        cx, cy = int(round(track.centroid[0])), int(round(track.centroid[1]))
        arrow_dy = int(round(max(-60.0, min(60.0, track.current_vy * 0.25))))
        arrow_dx = int(round(max(-60.0, min(60.0, track.current_vx * 0.25))))
        if abs(arrow_dy) >= 3 or abs(arrow_dx) >= 3:
            cv2.arrowedLine(
                canvas,
                (cx, cy),
                (cx + arrow_dx, cy + arrow_dy),
                box_color,
                2,
                tipLength=0.3,
            )

        # Build label text
        parts: List[str] = []
        if self._config.show_track_ids:
            parts.append(f"ID:{track.track_id}")
        parts.append(track.class_name)
        if self._config.show_confidence:
            parts.append(f"{track.confidence:.2f}")
        if self._config.show_velocity:
            parts.append(f"vy:{track.current_vy:.1f}px/s")
        parts.append(track.lifecycle_state.value)
        if risk is not None:
            parts.append(f"RISK:{risk.risk_level.value}({risk.overlap_percentage:.1f}%)")

        label = " | ".join(parts)
        cv2.putText(
            canvas,
            label,
            (max(5, x1), max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            risk_color,
            2,
            cv2.LINE_AA,
        )

    def _draw_hud(
        self,
        canvas: np.ndarray,
        fps: float,
        active_track_count: int,
        incident_count: int,
        baseline_complete: bool,
    ) -> None:
        """Render top-left telemetry HUD overlay."""
        cv2.rectangle(canvas, (8, 8), (360, 108), COLOR_DARK_BG, -1)
        status_str = "MONITORING" if baseline_complete else "PROFILING BASELINE"
        status_color = COLOR_GREEN if baseline_complete else COLOR_YELLOW

        cv2.putText(
            canvas,
            f"State: {status_str}",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_color,
            2,
            cv2.LINE_AA,
        )
        if self._config.show_fps:
            cv2.putText(
                canvas,
                f"FPS: {fps:.1f}",
                (16, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                COLOR_WHITE,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            f"Active Tracks: {active_track_count} | Incidents: {incident_count}",
            (16, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Keys: [q] Quit  [p] Pause  [s] Screenshot",
            (16, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            COLOR_GREY,
            1,
            cv2.LINE_AA,
        )
