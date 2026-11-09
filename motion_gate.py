"""Lightweight motion pre-association and downward velocity gating before ByteTrack."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Deque, Dict, List, Optional, Set, Tuple

from baseline_profiler import compute_bbox_iou, compute_euclidean_distance
from config_manager import MotionConfig
from detector import Detection


class LifecycleState(str, Enum):
    """Explicit object lifecycle states across motion gating, tracking, and risk evaluation."""

    UNSEEN = "UNSEEN"
    OBSERVED = "OBSERVED"
    MOTION_CANDIDATE = "MOTION_CANDIDATE"
    DESCENDING_CANDIDATE = "DESCENDING_CANDIDATE"
    FALLING = "FALLING"
    POSSIBLE_LANDING = "POSSIBLE_LANDING"
    LANDED = "LANDED"
    RISK_ASSESSED = "RISK_ASSESSED"
    INCIDENT_RECORDED = "INCIDENT_RECORDED"
    REJECTED = "REJECTED"
    TRACK_LOST = "TRACK_LOST"
    EXPIRED = "EXPIRED"
    LANDING_UNCERTAIN = "LANDING_UNCERTAIN"
    PROCESSING_FAILED = "PROCESSING_FAILED"


@dataclass
class PreAssociationCandidate:
    """Internal lightweight motion trace used prior to ByteTrack activation."""

    candidate_id: int
    class_id: int
    class_name: str
    initial_centroid: Tuple[float, float]
    current_centroid: Tuple[float, float]
    current_bbox: Tuple[float, float, float, float]
    first_timestamp: float
    last_timestamp: float
    observation_count: int
    missed_frames: int
    state: LifecycleState
    qualified_for_tracking: bool
    vy_history: Deque[float] = field(default_factory=deque)
    vx_history: Deque[float] = field(default_factory=deque)
    smoothed_vy: float = 0.0
    smoothed_vx: float = 0.0
    total_vertical_displacement: float = 0.0


@dataclass(frozen=True)
class EvaluatedDetection:
    """Detection enriched with pre-association motion metrics and lifecycle state."""

    detection: Detection
    candidate_id: int
    instantaneous_vy: float
    smoothed_vy: float
    smoothed_vx: float
    vertical_displacement: float
    state: LifecycleState
    qualifies_for_bytetrack: bool
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class MotionGateOutput:
    """Structured output returned by MotionGate to main.py."""

    evaluated_detections: List[EvaluatedDetection]
    qualifying_detections: List[EvaluatedDetection]
    rejected_detections: List[EvaluatedDetection]


class MotionGate:
    """
    Performs lightweight pre-association and downward velocity gating prior to ByteTrack.

    Calculates:
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        vy = (yc_current - yc_previous) / delta_time

    Positive vy corresponds to downward movement in image coordinates.
    Only detections exhibiting downward vertical velocity >= motion.descend_min_vy,
    downward displacement >= motion.minimum_displacement, and downward-dominant motion
    qualify for ByteTrack activation.
    """

    def __init__(self, config: MotionConfig, max_missed_frames: int = 5) -> None:
        self._config = config
        self._max_missed_frames = max(2, max_missed_frames)
        self._candidates: Dict[int, PreAssociationCandidate] = {}
        self._next_candidate_id: int = 1

    def evaluate(
        self,
        detections: List[Detection],
        timestamp: float,
        active_track_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> MotionGateOutput:
        """
        Associate current detections with lightweight motion candidates, calculate
        smoothed vertical velocity (pixels/second), and gate ByteTrack activation.
        """
        active_track_bboxes = active_track_bboxes or []
        evaluated: List[EvaluatedDetection] = []
        qualifying: List[EvaluatedDetection] = []
        rejected: List[EvaluatedDetection] = []

        matched_candidate_ids: Set[int] = set()

        for det in detections:
            xc = (det.bbox[0] + det.bbox[2]) / 2.0
            yc = (det.bbox[1] + det.bbox[3]) / 2.0

            cand = self._find_best_candidate(det, (xc, yc), matched_candidate_ids)

            if cand is None:
                cand_id = self._next_candidate_id
                self._next_candidate_id += 1
                cand = PreAssociationCandidate(
                    candidate_id=cand_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    initial_centroid=(xc, yc),
                    current_centroid=(xc, yc),
                    current_bbox=det.bbox,
                    first_timestamp=timestamp,
                    last_timestamp=timestamp,
                    observation_count=1,
                    missed_frames=0,
                    state=LifecycleState.OBSERVED,
                    qualified_for_tracking=False,
                    vy_history=deque(maxlen=self._config.velocity_smoothing_window),
                    vx_history=deque(maxlen=self._config.velocity_smoothing_window),
                )
                self._candidates[cand_id] = cand
                matched_candidate_ids.add(cand_id)

                # Check if this detection directly overlaps an already-active falling track
                matches_active_track = self._overlaps_active_track(det.bbox, active_track_bboxes)
                eval_det = EvaluatedDetection(
                    detection=det,
                    candidate_id=cand.candidate_id,
                    instantaneous_vy=0.0,
                    smoothed_vy=0.0,
                    smoothed_vx=0.0,
                    vertical_displacement=0.0,
                    state=LifecycleState.DESCENDING_CANDIDATE if matches_active_track else LifecycleState.OBSERVED,
                    qualifies_for_bytetrack=matches_active_track,
                    rejection_reason=None if matches_active_track else "initial_observation_awaiting_velocity",
                )
                evaluated.append(eval_det)
                if matches_active_track:
                    qualifying.append(eval_det)
                else:
                    rejected.append(eval_det)
                continue

            matched_candidate_ids.add(cand.candidate_id)
            dt = timestamp - cand.last_timestamp

            if dt <= 1e-6:
                inst_vy = cand.smoothed_vy
                inst_vx = cand.smoothed_vx
            else:
                inst_vy = (yc - cand.current_centroid[1]) / dt
                inst_vx = (xc - cand.current_centroid[0]) / dt

            cand.vy_history.append(inst_vy)
            cand.vx_history.append(inst_vx)
            cand.smoothed_vy = float(sum(cand.vy_history) / len(cand.vy_history))
            cand.smoothed_vx = float(sum(cand.vx_history) / len(cand.vx_history))
            cand.current_centroid = (xc, yc)
            cand.current_bbox = det.bbox
            cand.last_timestamp = timestamp
            cand.observation_count += 1
            cand.missed_frames = 0
            cand.total_vertical_displacement = yc - cand.initial_centroid[1]

            matches_active_track = self._overlaps_active_track(det.bbox, active_track_bboxes)
            qualifies, state, reason = self._classify_candidate_motion(cand, matches_active_track)
            cand.state = state
            if qualifies:
                cand.qualified_for_tracking = True

            eval_det = EvaluatedDetection(
                detection=det,
                candidate_id=cand.candidate_id,
                instantaneous_vy=inst_vy,
                smoothed_vy=cand.smoothed_vy,
                smoothed_vx=cand.smoothed_vx,
                vertical_displacement=cand.total_vertical_displacement,
                state=state,
                qualifies_for_bytetrack=qualifies,
                rejection_reason=reason,
            )
            evaluated.append(eval_det)
            if qualifies:
                qualifying.append(eval_det)
            else:
                rejected.append(eval_det)

        # Age unmatched candidates and prune expired ones
        for cid in list(self._candidates.keys()):
            if cid not in matched_candidate_ids:
                self._candidates[cid].missed_frames += 1
                if self._candidates[cid].missed_frames > self._max_missed_frames:
                    del self._candidates[cid]

        return MotionGateOutput(
            evaluated_detections=evaluated,
            qualifying_detections=qualifying,
            rejected_detections=rejected,
        )

    def _find_best_candidate(
        self,
        det: Detection,
        centroid: Tuple[float, float],
        excluded_ids: Set[int],
    ) -> Optional[PreAssociationCandidate]:
        """Match a detection to the most compatible prior motion candidate."""
        best_cand: Optional[PreAssociationCandidate] = None
        best_score = -1.0

        for cand in self._candidates.values():
            if cand.candidate_id in excluded_ids:
                continue
            if cand.class_id != det.class_id:
                continue

            iou = compute_bbox_iou(det.bbox, cand.current_bbox)
            dist = compute_euclidean_distance(centroid, cand.current_centroid)

            if (
                iou < self._config.preassociation_iou_threshold
                and dist > self._config.preassociation_distance_px
            ):
                continue

            dist_score = max(0.0, 1.0 - (dist / max(1.0, self._config.preassociation_distance_px)))
            score = iou * 0.6 + dist_score * 0.4
            if score > best_score:
                best_score = score
                best_cand = cand

        return best_cand

    def _classify_candidate_motion(
        self,
        cand: PreAssociationCandidate,
        matches_active_track: bool,
    ) -> Tuple[bool, LifecycleState, Optional[str]]:
        """
        Determine whether a pre-associated candidate qualifies for ByteTrack.

        Rejects:
        - Stationary objects
        - Upward-moving objects (vy < 0)
        - Purely horizontal objects
        - Insufficient displacement
        """
        if cand.qualified_for_tracking or matches_active_track:
            return True, LifecycleState.DESCENDING_CANDIDATE, None

        vy = cand.smoothed_vy
        vx = cand.smoothed_vx
        disp_y = cand.total_vertical_displacement
        total_disp = compute_euclidean_distance(cand.current_centroid, cand.initial_centroid)

        # Stationary rejection
        if abs(vy) < 1e-3 and total_disp < self._config.minimum_displacement:
            return False, LifecycleState.REJECTED, "stationary_object"

        # Upward motion rejection
        if vy < 0.0:
            return False, LifecycleState.REJECTED, "upward_motion"

        # Purely horizontal motion rejection
        if abs(vx) > abs(vy) * 1.5 and vy < self._config.descend_min_vy:
            return False, LifecycleState.REJECTED, "horizontal_motion"

        # Qualifying downward motion check
        if (
            vy >= self._config.descend_min_vy
            and disp_y >= self._config.minimum_displacement
            and abs(vy) >= abs(vx) * 0.5
        ):
            return True, LifecycleState.DESCENDING_CANDIDATE, None

        if vy > 0.0:
            return False, LifecycleState.MOTION_CANDIDATE, "insufficient_downward_velocity_or_displacement"

        return False, LifecycleState.REJECTED, "non_descending_motion"

    def _overlaps_active_track(
        self,
        bbox: Tuple[float, float, float, float],
        active_track_bboxes: List[Tuple[float, float, float, float]],
    ) -> bool:
        """Return True if bbox matches or is near an already active falling track."""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        for tb in active_track_bboxes:
            if compute_bbox_iou(bbox, tb) >= self._config.preassociation_iou_threshold:
                return True
            tcx = (tb[0] + tb[2]) / 2.0
            tcy = (tb[1] + tb[3]) / 2.0
            if math.hypot(cx - tcx, cy - tcy) <= self._config.preassociation_distance_px:
                return True
        return False
