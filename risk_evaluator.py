"""Shapely-based AMR corridor spatial intersection and hazard risk classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import math
from typing import Optional

from shapely.geometry import Polygon
from shapely.validation import explain_validity

from config_manager import (
    CalibrationConfig,
    ConfigManager,
    ConfigurationError,
    CorridorGeometry,
    RiskConfig,
)

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """AMR hazard risk classification levels."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class RiskAssessmentResult:
    """Structured output returned by RiskEvaluator to main.py."""

    risk_level: RiskLevel
    object_area: float
    intersection_area: float
    overlap_percentage: float
    touches_boundary: bool
    external_distance: float
    geometry_valid: bool
    justification: str
    intersection_polygon: Optional[Polygon] = None


class RiskEvaluator:
    """
    Evaluates the spatial relationship between a landed object's footprint
    polygon and the AMR navigation corridor using Shapely.

    Formulas:
        intersection = object_polygon.intersection(corridor_polygon)
        overlap_percentage = (intersection.area / object_polygon.area) * 100

    Classification Rules:
        - HIGH: overlap_percentage >= risk.high_overlap_threshold (default 10.0%)
        - MEDIUM: (0 < overlap_percentage < high_overlap_threshold) OR boundary-only contact
        - LOW: 0% overlap and positive distance from the corridor
        - UNCERTAIN: Invalid geometry, zero-area footprint, missing spatial info,
                     failed homography transformation, or invalid corridor.
    """

    def __init__(
        self,
        risk_config: RiskConfig,
        corridor: CorridorGeometry,
        calibration_config: Optional[CalibrationConfig] = None,
    ) -> None:
        self._risk_cfg = risk_config
        self._corridor = corridor
        self._calibration_cfg = calibration_config

    def evaluate(
        self,
        object_footprint: Optional[Polygon],
        corridor_contact_history: bool = False,
    ) -> RiskAssessmentResult:
        """
        Classify the final resting footprint by its overlap and distance from the corridor.
        Never classifies invalid or degenerate geometry as LOW risk.
        """
        eps = self._risk_cfg.geometry_epsilon

        # 1. Validate corridor polygon
        if (
            self._corridor is None
            or self._corridor.polygon is None
            or not isinstance(self._corridor.polygon, Polygon)
            or self._corridor.polygon.is_empty
            or not self._corridor.polygon.is_valid
            or self._corridor.polygon.area <= eps
        ):
            return RiskAssessmentResult(
                risk_level=RiskLevel.UNCERTAIN,
                object_area=0.0,
                intersection_area=0.0,
                overlap_percentage=0.0,
                touches_boundary=False,
                external_distance=0.0,
                geometry_valid=False,
                justification="Invalid or degenerate AMR corridor polygon.",
                intersection_polygon=None,
            )

        # 2. Validate object footprint presence and type
        if object_footprint is None or not isinstance(object_footprint, Polygon):
            return RiskAssessmentResult(
                risk_level=RiskLevel.UNCERTAIN,
                object_area=0.0,
                intersection_area=0.0,
                overlap_percentage=0.0,
                touches_boundary=False,
                external_distance=0.0,
                geometry_valid=False,
                justification="Missing or non-polygon object footprint.",
                intersection_polygon=None,
            )

        if object_footprint.is_empty or not object_footprint.is_valid:
            reason = explain_validity(object_footprint)
            return RiskAssessmentResult(
                risk_level=RiskLevel.UNCERTAIN,
                object_area=0.0,
                intersection_area=0.0,
                overlap_percentage=0.0,
                touches_boundary=False,
                external_distance=0.0,
                geometry_valid=False,
                justification=f"Invalid object footprint geometry: {reason}.",
                intersection_polygon=None,
            )

        #Check finite coordinates
        coords = list(object_footprint.exterior.coords)
        if not all(math.isfinite(c[0]) and math.isfinite(c[1]) for c in coords):
            return RiskAssessmentResult(
                risk_level=RiskLevel.UNCERTAIN,
                object_area=0.0,
                intersection_area=0.0,
                overlap_percentage=0.0,
                touches_boundary=False,
                external_distance=0.0,
                geometry_valid=False,
                justification="Object footprint contains non-finite coordinates.",
                intersection_polygon=None,
            )

        working_footprint = object_footprint

        # 3. Apply metric homography transformation when corridor is in metric mode
        if self._corridor.geometry_mode == "metric":
            if (
                self._calibration_cfg is None
                or not self._calibration_cfg.homography_enabled
                or self._calibration_cfg.homography_matrix is None
            ):
                return RiskAssessmentResult(
                    risk_level=RiskLevel.UNCERTAIN,
                    object_area=0.0,
                    intersection_area=0.0,
                    overlap_percentage=0.0,
                    touches_boundary=False,
                    external_distance=0.0,
                    geometry_valid=False,
                    justification="Metric geometry mode enabled without a valid homography matrix.",
                    intersection_polygon=None,
                )
            try:
                working_footprint = ConfigManager.transform_polygon_homography(
                    object_footprint, self._calibration_cfg.homography_matrix
                )
            except ConfigurationError as exc:
                return RiskAssessmentResult(
                    risk_level=RiskLevel.UNCERTAIN,
                    object_area=0.0,
                    intersection_area=0.0,
                    overlap_percentage=0.0,
                    touches_boundary=False,
                    external_distance=0.0,
                    geometry_valid=False,
                    justification=f"Failed homography transformation of footprint: {exc}",
                    intersection_polygon=None,
                )

        obj_area = float(working_footprint.area)
        if obj_area <= eps or not math.isfinite(obj_area):
            return RiskAssessmentResult(
                risk_level=RiskLevel.UNCERTAIN,
                object_area=0.0,
                intersection_area=0.0,
                overlap_percentage=0.0,
                touches_boundary=False,
                external_distance=0.0,
                geometry_valid=False,
                justification=f"Zero-area or near-zero object footprint (area={obj_area:.6e}).",
                intersection_polygon=None,
            )

        # 4. Perform Shapely intersection and distance calculations
        try:
            corridor_poly = self._corridor.polygon
            intersection = working_footprint.intersection(corridor_poly)
            inter_area = float(intersection.area)
            overlap_pct = (inter_area / obj_area) * 100.0
            overlap_pct = max(0.0, min(100.0, overlap_pct))

            touches_boundary = bool(
                working_footprint.touches(corridor_poly)
                or (inter_area <= eps and working_footprint.intersects(corridor_poly))
            )
            ext_distance = float(working_footprint.distance(corridor_poly))
        except Exception as exc:
            logger.error("Shapely spatial analysis failed: %s", exc)
            return RiskAssessmentResult(
                risk_level=RiskLevel.UNCERTAIN,
                object_area=obj_area,
                intersection_area=0.0,
                overlap_percentage=0.0,
                touches_boundary=False,
                external_distance=0.0,
                geometry_valid=False,
                justification=f"Spatial calculation error: {exc}",
                intersection_polygon=None,
            )

        inter_poly: Optional[Polygon] = None
        if isinstance(intersection, Polygon) and not intersection.is_empty and intersection.area > eps:
            inter_poly = intersection
        elif hasattr(intersection, "geoms"):
            polys = [
                g for g in intersection.geoms if isinstance(g, Polygon) and g.area > eps
            ]
            if polys:
                inter_poly = max(polys, key=lambda p: p.area)

        # 5. Classify Risk Level
        if overlap_pct >= self._risk_cfg.high_overlap_threshold:
            return RiskAssessmentResult(
                risk_level=RiskLevel.HIGH,
                object_area=obj_area,
                intersection_area=inter_area,
                overlap_percentage=round(overlap_pct, 4),
                touches_boundary=touches_boundary,
                external_distance=round(ext_distance, 4),
                geometry_valid=True,
                justification=(
                    f"The object overlaps {overlap_pct:.1f}% of its footprint with the AMR corridor "
                    f"(>= {self._risk_cfg.high_overlap_threshold:.1f}% HIGH threshold)."
                ),
                intersection_polygon=inter_poly,
            )

        if overlap_pct > 0.0:
            return RiskAssessmentResult(
                risk_level=RiskLevel.MEDIUM,
                object_area=obj_area,
                intersection_area=inter_area,
                overlap_percentage=round(overlap_pct, 4),
                touches_boundary=touches_boundary,
                external_distance=round(ext_distance, 4),
                geometry_valid=True,
                justification=(
                    f"The object overlaps {overlap_pct:.2f}% of its footprint with the AMR corridor "
                    f"(below {self._risk_cfg.high_overlap_threshold:.1f}% HIGH threshold)."
                ),
                intersection_polygon=inter_poly,
            )

        if touches_boundary:
            return RiskAssessmentResult(
                risk_level=RiskLevel.MEDIUM,
                object_area=obj_area,
                intersection_area=0.0,
                overlap_percentage=0.0,
                touches_boundary=True,
                external_distance=0.0,
                geometry_valid=True,
                justification="The object boundary touches the corridor margin; risk is MEDIUM.",
                intersection_polygon=None,
            )

        if inter_area <= eps and not touches_boundary and ext_distance > 0.0:
            unit_label = "m" if self._corridor.geometry_mode == "metric" else "px"
            return RiskAssessmentResult(
                risk_level=RiskLevel.LOW,
                object_area=obj_area,
                intersection_area=0.0,
                overlap_percentage=0.0,
                touches_boundary=False,
                external_distance=round(ext_distance, 4),
                geometry_valid=True,
                justification=(
                    f"The final footprint has 0.0% overlap and is {ext_distance:.2f} {unit_label} "
                    f"from the corridor; risk is LOW."
                ),
                intersection_polygon=None,
            )

        return RiskAssessmentResult(
            risk_level=RiskLevel.UNCERTAIN,
            object_area=obj_area,
            intersection_area=inter_area,
            overlap_percentage=round(overlap_pct, 4),
            touches_boundary=touches_boundary,
            external_distance=round(ext_distance, 4),
            geometry_valid=False,
            justification="Ambiguous spatial state: zero overlap and zero external distance without boundary touch.",
            intersection_polygon=None,
        )
