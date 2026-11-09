"""Configuration management, validation, and AMR corridor geometry construction."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.validation import explain_validity


class ConfigurationError(ValueError):
    """Raised when configuration keys are missing, malformed, or out of valid bounds."""


@dataclass(frozen=True)
class InputConfig:
    """Configuration parameters for video or Isaac Sim camera input."""

    source: str
    video_path: str
    isaac_image_topic: str
    isaac_encoding: str
    reconnect_enabled: bool
    reconnect_attempts: int


@dataclass(frozen=True)
class YoloConfig:
    """Configuration parameters for YOLO26 inference."""

    path: str
    conf_threshold: float
    iou_threshold: float
    device: str
    image_size: int


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration parameters for baseline stationary debris registration."""

    enabled: bool
    observation_duration_sec: float
    minimum_observations: int
    stationary_iou_threshold: float
    max_stationary_displacement_px: float


@dataclass(frozen=True)
class CorridorConfig:
    """Configuration parameters for the AMR navigation corridor."""

    yellow_line_p1: Tuple[float, float]
    yellow_line_p2: Tuple[float, float]
    geometry_mode: str


@dataclass(frozen=True)
class AmrConfig:
    """Configuration parameters for AMR physical width and safety margins."""

    base_half_width: float
    safety_margin: float
    base_half_width_m: float
    safety_margin_m: float


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration parameters for optional homography ground-plane calibration."""

    homography_enabled: bool
    homography_matrix: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass(frozen=True)
class MotionConfig:
    """Configuration parameters for lightweight pre-association and downfall motion gating."""

    descend_min_vy: float
    velocity_units: str
    minimum_displacement: float
    velocity_smoothing_window: int
    preassociation_iou_threshold: float
    preassociation_distance_px: float


@dataclass(frozen=True)
class TrackingConfig:
    """Configuration parameters for selective ByteTrack and persistent object tracking."""

    maximum_missed_frames: int
    minimum_track_observations: int
    track_match_iou: float
    track_match_distance_px: float
    maximum_active_tracks: int
    byte_track_high_threshold: float
    byte_track_low_threshold: float
    byte_track_match_threshold: float
    track_buffer_frames: int


@dataclass(frozen=True)
class LandingConfig:
    """Configuration parameters for landing verification."""

    rest_frames: int
    rest_max_velocity: float
    maximum_detection_gap_frames: int
    minimum_track_age_frames: int
    position_stability_threshold_px: float


@dataclass(frozen=True)
class RiskConfig:
    """Configuration parameters for spatial overlap and AMR risk classification."""

    high_overlap_threshold: float
    medium_overlap_min: float
    geometry_epsilon: float


@dataclass(frozen=True)
class StorageConfig:
    """Configuration parameters for incident snapshot, clip, and metadata storage."""

    evidence_base_dir: str
    buffer_duration_sec: float
    post_event_duration_sec: float
    camera_fps: int
    video_codec: str
    jpeg_quality: int
    disk_min_free_mb: int


@dataclass(frozen=True)
class DisplayConfig:
    """Configuration parameters for OpenCV visualization."""

    enabled: bool
    window_name: str
    show_fps: bool
    show_trajectories: bool
    show_confidence: bool
    show_track_ids: bool
    show_velocity: bool
    show_baseline_debris: bool


@dataclass(frozen=True)
class PerformanceConfig:
    """Configuration parameters for frame skipping and FPS throttling."""

    frame_skip: int
    max_processing_fps: float


@dataclass(frozen=True)
class CorridorGeometry:
    """Constructed AMR corridor geometry and metadata."""

    p1_pixel: Tuple[float, float]
    p2_pixel: Tuple[float, float]
    p1_working: Tuple[float, float]
    p2_working: Tuple[float, float]
    half_width_working: float
    normal_vector: Tuple[float, float]
    vertices_working: List[Tuple[float, float]]
    vertices_pixel: List[Tuple[float, float]]
    polygon: Polygon
    geometry_mode: str


@dataclass(frozen=True)
class AppConfig:
    """Aggregate application configuration container."""

    input: InputConfig
    model: YoloConfig
    baseline: BaselineConfig
    corridor: CorridorConfig
    amr: AmrConfig
    calibration: CalibrationConfig
    motion: MotionConfig
    tracking: TrackingConfig
    landing: LandingConfig
    risk: RiskConfig
    storage: StorageConfig
    display: DisplayConfig
    performance: PerformanceConfig


REQUIRED_KEYS: Tuple[str, ...] = (
    "input.source",
    "input.video_path",
    "input.isaac_image_topic",
    "input.isaac_encoding",
    "input.reconnect_enabled",
    "input.reconnect_attempts",
    "model.path",
    "model.conf_threshold",
    "model.iou_threshold",
    "model.device",
    "model.image_size",
    "baseline.enabled",
    "baseline.observation_duration_sec",
    "baseline.minimum_observations",
    "baseline.stationary_iou_threshold",
    "baseline.max_stationary_displacement_px",
    "corridor.yellow_line_p1",
    "corridor.yellow_line_p2",
    "corridor.geometry_mode",
    "amr.base_half_width",
    "amr.safety_margin",
    "amr.base_half_width_m",
    "amr.safety_margin_m",
    "calibration.homography_enabled",
    "calibration.homography_matrix",
    "motion.descend_min_vy",
    "motion.velocity_units",
    "motion.minimum_displacement",
    "motion.velocity_smoothing_window",
    "motion.preassociation_iou_threshold",
    "motion.preassociation_distance_px",
    "tracking.maximum_missed_frames",
    "tracking.minimum_track_observations",
    "tracking.track_match_iou",
    "tracking.track_match_distance_px",
    "tracking.maximum_active_tracks",
    "tracking.byte_track_high_threshold",
    "tracking.byte_track_low_threshold",
    "tracking.byte_track_match_threshold",
    "tracking.track_buffer_frames",
    "landing.rest_frames",
    "landing.rest_max_velocity",
    "landing.maximum_detection_gap_frames",
    "landing.minimum_track_age_frames",
    "landing.position_stability_threshold_px",
    "risk.high_overlap_threshold",
    "risk.medium_overlap_min",
    "risk.geometry_epsilon",
    "storage.evidence_base_dir",
    "storage.buffer_duration_sec",
    "storage.post_event_duration_sec",
    "storage.camera_fps",
    "storage.video_codec",
    "storage.jpeg_quality",
    "storage.disk_min_free_mb",
    "display.enabled",
    "display.window_name",
    "display.show_fps",
    "display.show_trajectories",
    "display.show_confidence",
    "display.show_track_ids",
    "display.show_velocity",
    "display.show_baseline_debris",
    "performance.frame_skip",
    "performance.max_processing_fps",
)


class ConfigManager:
    """Loads, validates, and manages operational parameters and corridor geometry."""

    @staticmethod
    def parse_properties_file(file_path: str | Path) -> Dict[str, str]:
        """Parse a Java-style .properties file into a dictionary of string key-values."""
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise ConfigurationError(f"Configuration file not found: {path}")

        props: Dict[str, str] = {}
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(f"Failed to read configuration file {path}: {exc}") from exc

        for line_num, raw_line in enumerate(content.splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("!"):
                continue
            if "=" not in stripped:
                raise ConfigurationError(
                    f"Malformed property on line {line_num} in {path}: '{raw_line}'"
                )
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise ConfigurationError(f"Empty key on line {line_num} in {path}")
            props[key] = value
        return props

    @staticmethod
    def parse_int(props: Dict[str, str], key: str, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
        """Parse and validate an integer property."""
        raw = props.get(key, "")
        try:
            val = int(raw)
        except ValueError as exc:
            raise ConfigurationError(f"Property '{key}' must be an integer, got '{raw}'") from exc
        if min_val is not None and val < min_val:
            raise ConfigurationError(f"Property '{key}' must be >= {min_val}, got {val}")
        if max_val is not None and val > max_val:
            raise ConfigurationError(f"Property '{key}' must be <= {max_val}, got {val}")
        return val

    @staticmethod
    def parse_float(
        props: Dict[str, str],
        key: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        strictly_positive: bool = False,
    ) -> float:
        """Parse and validate a floating-point property."""
        raw = props.get(key, "")
        try:
            val = float(raw)
        except ValueError as exc:
            raise ConfigurationError(f"Property '{key}' must be a float, got '{raw}'") from exc
        if not math.isfinite(val):
            raise ConfigurationError(f"Property '{key}' must be finite, got '{raw}'")
        if strictly_positive and val <= 0.0:
            raise ConfigurationError(f"Property '{key}' must be > 0.0, got {val}")
        if min_val is not None and val < min_val:
            raise ConfigurationError(f"Property '{key}' must be >= {min_val}, got {val}")
        if max_val is not None and val > max_val:
            raise ConfigurationError(f"Property '{key}' must be <= {max_val}, got {val}")
        return val

    @staticmethod
    def parse_bool(props: Dict[str, str], key: str) -> bool:
        """Parse and validate a boolean property."""
        raw = props.get(key, "").strip().lower()
        if raw in ("true", "1", "yes", "on"):
            return True
        if raw in ("false", "0", "no", "off"):
            return False
        raise ConfigurationError(
            f"Property '{key}' must be a boolean (true/false), got '{props.get(key, '')}'"
        )

    @staticmethod
    def parse_coordinate(props: Dict[str, str], key: str) -> Tuple[float, float]:
        """Parse a 2D coordinate formatted as 'x,y'."""
        raw = props.get(key, "").strip()
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 2:
            raise ConfigurationError(
                f"Property '{key}' must be a 2D coordinate 'x,y', got '{raw}'"
            )
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError as exc:
            raise ConfigurationError(
                f"Property '{key}' contains non-numeric coordinates: '{raw}'"
            ) from exc
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ConfigurationError(f"Property '{key}' coordinates must be finite: '{raw}'")
        return (x, y)

    @staticmethod
    def parse_homography(
        props: Dict[str, str], enabled: bool, key: str = "calibration.homography_matrix"
    ) -> Optional[np.ndarray]:
        """Parse and validate an optional 3x3 homography matrix."""
        raw = props.get(key, "").strip()
        if not raw:
            if enabled:
                raise ConfigurationError(
                    "Homography is enabled ('calibration.homography_enabled=true') "
                    "but 'calibration.homography_matrix' is empty."
                )
            return None

        parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
        if len(parts) != 9:
            raise ConfigurationError(
                f"Property '{key}' must contain exactly 9 comma-separated floats for a 3x3 matrix, got {len(parts)}"
            )
        try:
            values = [float(x) for x in parts]
        except ValueError as exc:
            raise ConfigurationError(f"Property '{key}' contains non-float entries: '{raw}'") from exc
        if not all(math.isfinite(v) for v in values):
            raise ConfigurationError(f"Property '{key}' contains non-finite values.")

        matrix = np.array(values, dtype=np.float64).reshape((3, 3))
        det = float(np.linalg.det(matrix))
        if abs(det) < 1e-12:
            raise ConfigurationError(
                f"Homography matrix in '{key}' is singular or near-singular (det={det:.3e})."
            )
        return matrix

    @classmethod
    def load(
        cls,
        config_path: str | Path,
        overrides: Optional[Dict[str, str]] = None,
    ) -> AppConfig:
        """Load, override, and validate the complete application configuration."""
        props = cls.parse_properties_file(config_path)
        if overrides:
            for k, v in overrides.items():
                if v is not None:
                    props[k] = str(v)

        missing = [k for k in REQUIRED_KEYS if k not in props]
        if missing:
            raise ConfigurationError(
                f"Missing required configuration keys: {', '.join(missing)}"
            )

        source = props["input.source"].strip().lower()
        if source not in ("video", "isaac"):
            raise ConfigurationError(
                f"Property 'input.source' must be 'video' or 'isaac', got '{source}'"
            )

        video_path = props["input.video_path"].strip()
        if not video_path:
            raise ConfigurationError("Property 'input.video_path' cannot be empty.")

        isaac_topic = props["input.isaac_image_topic"].strip()
        if not isaac_topic.startswith("/"):
            raise ConfigurationError(
                f"Property 'input.isaac_image_topic' must be a valid ROS topic starting with '/', got '{isaac_topic}'"
            )

        isaac_encoding = props["input.isaac_encoding"].strip().lower()
        if isaac_encoding not in ("bgr8", "rgb8", "rgba8", "bgra8", "mono8"):
            raise ConfigurationError(
                f"Unsupported 'input.isaac_encoding': '{isaac_encoding}'"
            )

        input_cfg = InputConfig(
            source=source,
            video_path=video_path,
            isaac_image_topic=isaac_topic,
            isaac_encoding=isaac_encoding,
            reconnect_enabled=cls.parse_bool(props, "input.reconnect_enabled"),
            reconnect_attempts=cls.parse_int(props, "input.reconnect_attempts", min_val=0, max_val=100),
        )

        model_path = props["model.path"].strip()
        if not model_path:
            raise ConfigurationError("Property 'model.path' cannot be empty.")

        yolo_cfg = YoloConfig(
            path=model_path,
            conf_threshold=cls.parse_float(props, "model.conf_threshold", min_val=0.01, max_val=1.0),
            iou_threshold=cls.parse_float(props, "model.iou_threshold", min_val=0.01, max_val=1.0),
            device=props["model.device"].strip(),
            image_size=cls.parse_int(props, "model.image_size", min_val=64, max_val=4096),
        )

        baseline_cfg = BaselineConfig(
            enabled=cls.parse_bool(props, "baseline.enabled"),
            observation_duration_sec=cls.parse_float(
                props, "baseline.observation_duration_sec", min_val=0.0, max_val=3600.0
            ),
            minimum_observations=cls.parse_int(
                props, "baseline.minimum_observations", min_val=1, max_val=10000
            ),
            stationary_iou_threshold=cls.parse_float(
                props, "baseline.stationary_iou_threshold", min_val=0.05, max_val=1.0
            ),
            max_stationary_displacement_px=cls.parse_float(
                props, "baseline.max_stationary_displacement_px", min_val=0.0, max_val=1000.0
            ),
        )

        p1 = cls.parse_coordinate(props, "corridor.yellow_line_p1")
        p2 = cls.parse_coordinate(props, "corridor.yellow_line_p2")
        geometry_mode = props["corridor.geometry_mode"].strip().lower()
        if geometry_mode not in ("pixel", "metric"):
            raise ConfigurationError(
                f"Property 'corridor.geometry_mode' must be 'pixel' or 'metric', got '{geometry_mode}'"
            )

        corridor_cfg = CorridorConfig(
            yellow_line_p1=p1,
            yellow_line_p2=p2,
            geometry_mode=geometry_mode,
        )

        amr_cfg = AmrConfig(
            base_half_width=cls.parse_float(props, "amr.base_half_width", strictly_positive=True),
            safety_margin=cls.parse_float(props, "amr.safety_margin", min_val=0.0),
            base_half_width_m=cls.parse_float(props, "amr.base_half_width_m", strictly_positive=True),
            safety_margin_m=cls.parse_float(props, "amr.safety_margin_m", min_val=0.0),
        )

        homography_enabled = cls.parse_bool(props, "calibration.homography_enabled")
        homography_matrix = cls.parse_homography(props, homography_enabled)
        if geometry_mode == "metric" and (not homography_enabled or homography_matrix is None):
            raise ConfigurationError(
                "corridor.geometry_mode='metric' requires 'calibration.homography_enabled=true' "
                "and a valid 3x3 'calibration.homography_matrix'."
            )

        calibration_cfg = CalibrationConfig(
            homography_enabled=homography_enabled,
            homography_matrix=homography_matrix,
        )

        velocity_units = props["motion.velocity_units"].strip().lower()
        if velocity_units != "pixels_per_second":
            raise ConfigurationError(
                f"Property 'motion.velocity_units' must be 'pixels_per_second', got '{velocity_units}'"
            )

        motion_cfg = MotionConfig(
            descend_min_vy=cls.parse_float(props, "motion.descend_min_vy", strictly_positive=True),
            velocity_units=velocity_units,
            minimum_displacement=cls.parse_float(props, "motion.minimum_displacement", strictly_positive=True),
            velocity_smoothing_window=cls.parse_int(
                props, "motion.velocity_smoothing_window", min_val=1, max_val=200
            ),
            preassociation_iou_threshold=cls.parse_float(
                props, "motion.preassociation_iou_threshold", min_val=0.0, max_val=1.0
            ),
            preassociation_distance_px=cls.parse_float(
                props, "motion.preassociation_distance_px", strictly_positive=True
            ),
        )

        high_thresh = cls.parse_float(props, "tracking.byte_track_high_threshold", min_val=0.01, max_val=1.0)
        low_thresh = cls.parse_float(props, "tracking.byte_track_low_threshold", min_val=0.0, max_val=1.0)
        if low_thresh > high_thresh:
            raise ConfigurationError(
                f"'tracking.byte_track_low_threshold' ({low_thresh}) cannot exceed "
                f"'tracking.byte_track_high_threshold' ({high_thresh})."
            )

        tracking_cfg = TrackingConfig(
            maximum_missed_frames=cls.parse_int(props, "tracking.maximum_missed_frames", min_val=1, max_val=1000),
            minimum_track_observations=cls.parse_int(
                props, "tracking.minimum_track_observations", min_val=1, max_val=500
            ),
            track_match_iou=cls.parse_float(props, "tracking.track_match_iou", min_val=0.01, max_val=1.0),
            track_match_distance_px=cls.parse_float(
                props, "tracking.track_match_distance_px", strictly_positive=True
            ),
            maximum_active_tracks=cls.parse_int(props, "tracking.maximum_active_tracks", min_val=1, max_val=10000),
            byte_track_high_threshold=high_thresh,
            byte_track_low_threshold=low_thresh,
            byte_track_match_threshold=cls.parse_float(
                props, "tracking.byte_track_match_threshold", min_val=0.01, max_val=1.0
            ),
            track_buffer_frames=cls.parse_int(props, "tracking.track_buffer_frames", min_val=1, max_val=1000),
        )

        landing_cfg = LandingConfig(
            rest_frames=cls.parse_int(props, "landing.rest_frames", min_val=1, max_val=1000),
            rest_max_velocity=cls.parse_float(props, "landing.rest_max_velocity", min_val=0.0, max_val=500.0),
            maximum_detection_gap_frames=cls.parse_int(
                props, "landing.maximum_detection_gap_frames", min_val=0, max_val=100
            ),
            minimum_track_age_frames=cls.parse_int(
                props, "landing.minimum_track_age_frames", min_val=1, max_val=1000
            ),
            position_stability_threshold_px=cls.parse_float(
                props, "landing.position_stability_threshold_px", strictly_positive=True
            ),
        )

        high_overlap = cls.parse_float(props, "risk.high_overlap_threshold", strictly_positive=True, max_val=100.0)
        medium_min = cls.parse_float(props, "risk.medium_overlap_min", min_val=0.0, max_val=100.0)
        if medium_min >= high_overlap:
            raise ConfigurationError(
                f"'risk.medium_overlap_min' ({medium_min}) must be strictly less than "
                f"'risk.high_overlap_threshold' ({high_overlap})."
            )

        risk_cfg = RiskConfig(
            high_overlap_threshold=high_overlap,
            medium_overlap_min=medium_min,
            geometry_epsilon=cls.parse_float(props, "risk.geometry_epsilon", strictly_positive=True, max_val=1.0),
        )

        codec = props["storage.video_codec"].strip()
        if len(codec) != 4:
            raise ConfigurationError(
                f"Property 'storage.video_codec' must be a 4-character FOURCC code, got '{codec}'"
            )

        storage_cfg = StorageConfig(
            evidence_base_dir=props["storage.evidence_base_dir"].strip(),
            buffer_duration_sec=cls.parse_float(props, "storage.buffer_duration_sec", strictly_positive=True),
            post_event_duration_sec=cls.parse_float(props, "storage.post_event_duration_sec", strictly_positive=True),
            camera_fps=cls.parse_int(props, "storage.camera_fps", min_val=1, max_val=240),
            video_codec=codec,
            jpeg_quality=cls.parse_int(props, "storage.jpeg_quality", min_val=1, max_val=100),
            disk_min_free_mb=cls.parse_int(props, "storage.disk_min_free_mb", min_val=1),
        )

        display_cfg = DisplayConfig(
            enabled=cls.parse_bool(props, "display.enabled"),
            window_name=props["display.window_name"].strip() or "AMR Downfall Monitoring",
            show_fps=cls.parse_bool(props, "display.show_fps"),
            show_trajectories=cls.parse_bool(props, "display.show_trajectories"),
            show_confidence=cls.parse_bool(props, "display.show_confidence"),
            show_track_ids=cls.parse_bool(props, "display.show_track_ids"),
            show_velocity=cls.parse_bool(props, "display.show_velocity"),
            show_baseline_debris=cls.parse_bool(props, "display.show_baseline_debris"),
        )

        performance_cfg = PerformanceConfig(
            frame_skip=cls.parse_int(props, "performance.frame_skip", min_val=0, max_val=100),
            max_processing_fps=cls.parse_float(props, "performance.max_processing_fps", min_val=0.0, max_val=1000.0),
        )

        app_cfg = AppConfig(
            input=input_cfg,
            model=yolo_cfg,
            baseline=baseline_cfg,
            corridor=corridor_cfg,
            amr=amr_cfg,
            calibration=calibration_cfg,
            motion=motion_cfg,
            tracking=tracking_cfg,
            landing=landing_cfg,
            risk=risk_cfg,
            storage=storage_cfg,
            display=display_cfg,
            performance=performance_cfg,
        )

        # Validate corridor construction eagerly so invalid centerlines fail fast
        cls.build_amr_corridor(app_cfg)
        return app_cfg

    @staticmethod
    def transform_point_homography(
        point: Tuple[float, float], matrix: np.ndarray
    ) -> Tuple[float, float]:
        """Apply a 3x3 projective homography matrix to a 2D coordinate."""
        vec = np.array([point[0], point[1], 1.0], dtype=np.float64)
        projected = matrix @ vec
        w = float(projected[2])
        if abs(w) < 1e-12:
            raise ConfigurationError("Homography transformation resulted in division by zero (w ~ 0).")
        x = float(projected[0] / w)
        y = float(projected[1] / w)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ConfigurationError("Homography transformation produced non-finite coordinates.")
        return (x, y)

    @classmethod
    def transform_polygon_homography(cls, polygon: Polygon, matrix: np.ndarray) -> Polygon:
        """Transform a Shapely Polygon using a 3x3 homography matrix."""
        if polygon.is_empty:
            raise ConfigurationError("Cannot transform an empty polygon.")
        coords = list(polygon.exterior.coords)
        transformed_coords = [cls.transform_point_homography(( float(pt[0]), float(pt[1]) ), matrix) for pt in coords]
        transformed_poly = Polygon(transformed_coords)
        if not transformed_poly.is_valid or transformed_poly.area <= 0.0:
            raise ConfigurationError(
                f"Transformed polygon is invalid: {explain_validity(transformed_poly)}"
            )
        return transformed_poly

    @classmethod
    def build_amr_corridor(cls, config: AppConfig) -> CorridorGeometry:
        """
        Construct the AMR navigation corridor polygon from P1, P2, and configured widths.

        Formulas:
            dx = x2 - x1
            dy = y2 - y1
            L = sqrt(dx^2 + dy^2)
            nx = -dy / L
            ny = dx / L
            W = amr.base_half_width + amr.safety_margin (or metric equivalents)
            Vertices:
                P1 + normal * W
                P2 + normal * W
                P2 - normal * W
                P1 - normal * W
        """
        p1_px = config.corridor.yellow_line_p1
        p2_px = config.corridor.yellow_line_p2

        if config.corridor.geometry_mode == "metric":
            if not config.calibration.homography_enabled or config.calibration.homography_matrix is None:
                raise ConfigurationError(
                    "Metric corridor mode requires enabled homography and valid homography matrix."
                )
            p1_work = cls.transform_point_homography(p1_px, config.calibration.homography_matrix)
            p2_work = cls.transform_point_homography(p2_px, config.calibration.homography_matrix)
            width_w = config.amr.base_half_width_m + config.amr.safety_margin_m
        else:
            p1_work = p1_px
            p2_work = p2_px
            width_w = config.amr.base_half_width + config.amr.safety_margin

        if width_w <= 0.0 or not math.isfinite(width_w):
            raise ConfigurationError(f"Invalid total corridor half-width W={width_w}")

        x1, y1 = p1_work
        x2, y2 = p2_work
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)

        if length <= config.risk.geometry_epsilon:
            raise ConfigurationError(
                f"Zero-length or degenerate corridor centerline: P1={p1_work}, P2={p2_work}, L={length}"
            )

        nx = -dy / length
        ny = dx / length

        v1 = (x1 + nx * width_w, y1 + ny * width_w)
        v2 = (x2 + nx * width_w, y2 + ny * width_w)
        v3 = (x2 - nx * width_w, y2 - ny * width_w)
        v4 = (x1 - nx * width_w, y1 - ny * width_w)
        vertices_work = [v1, v2, v3, v4]

        # Also compute pixel-space vertices for visualization
        dx_px = p2_px[0] - p1_px[0]
        dy_px = p2_px[1] - p1_px[1]
        len_px = math.sqrt(dx_px * dx_px + dy_px * dy_px)
        if len_px <= config.risk.geometry_epsilon:
            raise ConfigurationError("Zero-length pixel centerline.")
        nx_px = -dy_px / len_px
        ny_px = dx_px / len_px
        w_px = config.amr.base_half_width + config.amr.safety_margin
        vertices_px = [
            (p1_px[0] + nx_px * w_px, p1_px[1] + ny_px * w_px),
            (p2_px[0] + nx_px * w_px, p2_px[1] + ny_px * w_px),
            (p2_px[0] - nx_px * w_px, p2_px[1] - ny_px * w_px),
            (p1_px[0] - nx_px * w_px, p1_px[1] - ny_px * w_px),
        ]

        polygon = Polygon(vertices_work)
        if not polygon.is_valid or polygon.is_empty or polygon.area <= config.risk.geometry_epsilon:
            raise ConfigurationError(
                f"Constructed AMR corridor polygon is invalid: {explain_validity(polygon)}, area={polygon.area}"
            )

        return CorridorGeometry(
            p1_pixel=p1_px,
            p2_pixel=p2_px,
            p1_working=p1_work,
            p2_working=p2_work,
            half_width_working=width_w,
            normal_vector=(nx, ny),
            vertices_working=vertices_work,
            vertices_pixel=vertices_px,
            polygon=polygon,
            geometry_mode=config.corridor.geometry_mode,
        )
