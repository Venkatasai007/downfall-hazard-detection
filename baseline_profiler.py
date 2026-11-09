"""Static baseline scene profiler for registering pre-existing stationary debris."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

from shapely.geometry import Polygon

from config_manager import BaselineConfig
from detector import Detection


def compute_bbox_iou(
    bbox_a: Tuple[float, float, float, float],
    bbox_b: Tuple[float, float, float, float],
) -> float:
    """Compute 2D Intersection-over-Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 1e-9:
        return 0.0
    return float(inter_area / union_area)


def compute_euclidean_distance(
    pt_a: Tuple[float, float], pt_b: Tuple[float, float]
) -> float:
    """Compute Euclidean distance between two 2D points."""
    return math.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1])


@dataclass
class BaselineEntry:
    """Internal record tracking a stationary candidate or confirmed pre-existing debris."""

    debris_id: int
    class_id: int
    class_name: str
    label: str
    bbox: Tuple[float, float, float, float]
    initial_centroid: Tuple[float, float]
    centroid: Tuple[float, float]
    footprint: Polygon
    last_iou: float
    observation_count: int
    max_displacement_px: float
    is_confirmed_debris: bool
    is_displaced: bool
    first_seen_timestamp: float
    last_seen_timestamp: float
    centroid_history: List[Tuple[float, float]] = field(default_factory=list)


@dataclass(frozen=True)
class BaselineUpdateResult:
    """Structured result returned to main.py after baseline processing of a frame."""

    baseline_complete: bool
    active_debris: List[BaselineEntry]
    non_debris_detections: List[Detection]
    preexisting_detection_ids: List[int]


class BaselineProfiler:
    """
    Establishes a static scene baseline before normal incident generation.

    Tracks object class, bounding box, centroid, IoU, observation count,
    position stability, and footprint across a configurable observation window.
    Confirmed stationary objects are registered as 'UNKNOWN_DEBRIS' and excluded
    from triggering falling-object incidents unless they subsequently move.
    """

    def __init__(self, config: BaselineConfig) -> None:
        self._config = config
        self._entries: Dict[int, BaselineEntry] = {}
        self._next_id: int = 1
        self._start_timestamp: Optional[float] = None
        self._baseline_complete: bool = not config.enabled

    @property
    def is_baseline_complete(self) -> bool:
        """Return True if the baseline observation window has completed."""
        return self._baseline_complete

    def get_registered_debris(self) -> List[BaselineEntry]:
        """Return all currently confirmed and undisplaced 'UNKNOWN_DEBRIS' records."""
        return [
            entry
            for entry in self._entries.values()
            if entry.is_confirmed_debris and not entry.is_displaced
        ]

    def update(self, detections: List[Detection], timestamp: float) -> BaselineUpdateResult:
        """
        Process frame detections against the baseline registry.

        During the observation window, accumulates observations of stationary objects.
        Once the window completes (and during normal monitoring), filters out stationary
        pre-existing debris while monitoring registered debris for subsequent displacement.
        """
        if not self._config.enabled:
            self._baseline_complete = True
            return BaselineUpdateResult(
                baseline_complete=True,
                active_debris=[],
                non_debris_detections=list(detections),
                preexisting_detection_ids=[],
            )

        if self._start_timestamp is None:
            self._start_timestamp = timestamp

        elapsed = max(0.0, timestamp - self._start_timestamp)

        if not self._baseline_complete:
            self._accumulate_baseline_observations(detections, timestamp)
            if elapsed >= self._config.observation_duration_sec:
                self._finalize_baseline_window()
                self._baseline_complete = True

        non_debris: List[Detection] = []
        preexisting_ids: List[int] = []
        matched_debris_ids: set[int] = set()

        for det in detections:
            matched_entry = self._match_existing_debris(det, matched_debris_ids)
            if matched_entry is not None:
                matched_debris_ids.add(matched_entry.debris_id)
                displacement = compute_euclidean_distance(
                    det.centroid, matched_entry.initial_centroid
                )
                iou = compute_bbox_iou(det.bbox, matched_entry.bbox)

                # Check if previously stationary baseline debris has started moving
                if (
                    displacement > self._config.max_stationary_displacement_px * 2.5
                    or iou < self._config.stationary_iou_threshold * 0.6
                ):
                    matched_entry.is_displaced = True
                    matched_entry.last_seen_timestamp = timestamp
                    non_debris.append(det)
                else:
                    matched_entry.bbox = det.bbox
                    matched_entry.centroid = det.centroid
                    matched_entry.footprint = det.footprint
                    matched_entry.last_iou = iou
                    matched_entry.observation_count += 1
                    matched_entry.max_displacement_px = max(
                        matched_entry.max_displacement_px, displacement
                    )
                    matched_entry.last_seen_timestamp = timestamp
                    preexisting_ids.append(det.detection_id)
            else:
                if self.is_preexisting_debris(det, threshold=self._config.stationary_iou_threshold):
                    preexisting_ids.append(det.detection_id)
                else:
                    non_debris.append(det)

        return BaselineUpdateResult(
            baseline_complete=self._baseline_complete,
            active_debris=self.get_registered_debris(),
            non_debris_detections=non_debris,
            preexisting_detection_ids=preexisting_ids,
        )

    def is_preexisting_debris(self, current_detection: Detection, threshold: float = 0.7) -> bool:
        """
        Return True if current_detection matches a confirmed, undisplaced baseline debris object
        with IoU >= threshold and stable position within configured displacement bounds.
        """
        if not self._config.enabled:
            return False

        for entry in self._entries.values():
            if not entry.is_confirmed_debris or entry.is_displaced:
                continue
            iou = compute_bbox_iou(current_detection.bbox, entry.bbox)
            displacement = compute_euclidean_distance(
                current_detection.centroid, entry.initial_centroid
            )
            if (
                iou >= threshold
                and displacement <= self._config.max_stationary_displacement_px * 2.0
            ):
                return True
        return False

    def _accumulate_baseline_observations(
        self, detections: List[Detection], timestamp: float
    ) -> None:
        """Update candidate baseline objects during the initial observation window."""
        used_entry_ids: set[int] = set()

        for det in detections:
            best_entry: Optional[BaselineEntry] = None
            best_iou = 0.0

            for entry in self._entries.values():
                if entry.debris_id in used_entry_ids or entry.is_displaced:
                    continue
                iou = compute_bbox_iou(det.bbox, entry.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_entry = entry

            if best_entry is not None and best_iou >= self._config.stationary_iou_threshold:
                used_entry_ids.add(best_entry.debris_id)
                disp = compute_euclidean_distance(det.centroid, best_entry.initial_centroid)
                best_entry.max_displacement_px = max(best_entry.max_displacement_px, disp)
                best_entry.last_iou = best_iou
                best_entry.bbox = det.bbox
                best_entry.centroid = det.centroid
                best_entry.footprint = det.footprint
                best_entry.last_seen_timestamp = timestamp
                best_entry.observation_count += 1
                best_entry.centroid_history.append(det.centroid)

                if disp > self._config.max_stationary_displacement_px:
                    best_entry.is_displaced = True
                    best_entry.is_confirmed_debris = False
                elif best_entry.observation_count >= self._config.minimum_observations:
                    best_entry.is_confirmed_debris = True
                    best_entry.label = "UNKNOWN_DEBRIS"
            else:
                new_id = self._next_id
                self._next_id += 1
                self._entries[new_id] = BaselineEntry(
                    debris_id=new_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    label="CANDIDATE_DEBRIS",
                    bbox=det.bbox,
                    initial_centroid=det.centroid,
                    centroid=det.centroid,
                    footprint=det.footprint,
                    last_iou=1.0,
                    observation_count=1,
                    max_displacement_px=0.0,
                    is_confirmed_debris=False,
                    is_displaced=False,
                    first_seen_timestamp=timestamp,
                    last_seen_timestamp=timestamp,
                    centroid_history=[det.centroid],
                )

    def _finalize_baseline_window(self) -> None:
        """Promote eligible stationary candidates to UNKNOWN_DEBRIS at the end of the window."""
        for entry in list(self._entries.values()):
            if (
                not entry.is_displaced
                and entry.observation_count >= self._config.minimum_observations
                and entry.max_displacement_px <= self._config.max_stationary_displacement_px
            ):
                entry.is_confirmed_debris = True
                entry.label = "UNKNOWN_DEBRIS"
            elif not entry.is_confirmed_debris:
                del self._entries[entry.debris_id]

    def _match_existing_debris(
        self, det: Detection, excluded_ids: set[int]
    ) -> Optional[BaselineEntry]:
        """Find the best-matching confirmed debris entry for a current detection."""
        best_entry: Optional[BaselineEntry] = None
        best_score = 0.0

        for entry in self._entries.values():
            if not entry.is_confirmed_debris or entry.is_displaced:
                continue
            if entry.debris_id in excluded_ids:
                continue

            iou = compute_bbox_iou(det.bbox, entry.bbox)
            dist = compute_euclidean_distance(det.centroid, entry.centroid)
            if iou >= self._config.stationary_iou_threshold * 0.5 or (
                entry.class_id == det.class_id
                and dist <= self._config.max_stationary_displacement_px * 3.0
            ):
                score = iou + max(0.0, 1.0 - dist / 50.0) * 0.1
                if score > best_score:
                    best_score = score
                    best_entry = entry

        return best_entry
