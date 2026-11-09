"""
Central orchestrator for Real-Time Downfall Hazard Detection and AMR Risk Segregation.

Pipeline:
    FRAME ACQUISITION -> FRAME VALIDATION -> UPDATE CIRCULAR BUFFER ->
    YOLO INFERENCE -> BASELINE REGISTRATION -> PRE-EXISTING DEBRIS FILTER ->
    LIGHTWEIGHT MOTION ASSOCIATION -> VERTICAL VELOCITY CALCULATION ->
    DOWNFALL GATING -> SELECTIVE BYTETRACK -> PERSISTENT TRACK MANAGEMENT ->
    VELOCITY / ACCELERATION -> LANDING VERIFICATION -> FINAL FOOTPRINT EXTRACTION ->
    AMR CORRIDOR INTERSECTION -> RISK CLASSIFICATION -> INCIDENT GENERATION ->
    PRE + POST EVENT VIDEO -> SNAPSHOT + JSON -> OPENCV VISUALIZATION -> NEXT FRAME
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple
import uuid

import cv2
import numpy as np
from shapely.geometry import Polygon

from baseline_profiler import BaselineProfiler, BaselineUpdateResult
from config_manager import (
    AppConfig,
    ConfigManager,
    ConfigurationError,
    CorridorGeometry,
)
from detector import Detection, DetectionFrameOutput, DetectorError, ObjectDetector
from evidence_logger import CompletedIncidentRecord, EvidenceLogger
from motion_gate import LifecycleState, MotionGate, MotionGateOutput
from risk_evaluator import RiskAssessmentResult, RiskEvaluator, RiskLevel
from track_manager import TrackedObject, TrackingFrameOutput, TrackManager
from visualization import VisualizationCommand, Visualizer

logger = logging.getLogger("DownfallHazardApp")


class FrameSourceError(RuntimeError):
    """Raised when a camera or video frame source fails to initialize or read."""


@dataclass(frozen=True)
class AcquiredFrame:
    """Common container returned by both Video and Isaac Sim ROS 2 frame sources."""

    frame_id: int
    timestamp: float
    frame: np.ndarray
    width: int
    height: int
    source_fps: float


@dataclass
class PerformanceTelemetry:
    """Runtime latency and throughput telemetry metrics."""

    yolo_latency_ms: float = 0.0
    tracking_latency_ms: float = 0.0
    geometry_latency_ms: float = 0.0
    total_frame_latency_ms: float = 0.0
    effective_fps: float = 0.0
    dropped_frames: int = 0
    processed_frames: int = 0
    active_tracks: int = 0
    pending_evidence_jobs: int = 0
    incidents_generated: int = 0


@dataclass(frozen=True)
class FrameProcessingResult:
    """Structured result of executing process_frame() on a single frame."""

    frame_id: int
    timestamp: float
    detections_count: int
    non_debris_count: int
    qualifying_motion_count: int
    active_tracks_count: int
    newly_landed_track_ids: List[int]
    triggered_incident_ids: List[str]
    risk_results: Dict[int, RiskAssessmentResult]
    rendered_frame: np.ndarray
    telemetry: PerformanceTelemetry


class FrameSource:
    """Abstract base interface for video and Isaac Sim ROS 2 frame acquisition."""

    def open(self) -> None:
        """Open and validate the underlying stream."""
        raise NotImplementedError

    def read(self) -> Tuple[bool, Optional[AcquiredFrame], bool]:
        """
        Read the next frame.
        Returns (success, AcquiredFrame or None, is_end_of_stream).
        """
        raise NotImplementedError

    def get_fps(self) -> float:
        """Return source frame rate in frames per second."""
        raise NotImplementedError

    def close(self) -> None:
        """Safely release underlying stream resources."""
        raise NotImplementedError


class VideoFrameSource(FrameSource):
    """Pre-recorded video file frame source backed by OpenCV VideoCapture."""

    def __init__(self, video_path: str, default_fps: int = 30) -> None:
        self._video_path = Path(video_path)
        self._default_fps = float(max(1, default_fps))
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = self._default_fps
        self._width: int = 0
        self._height: int = 0
        self._frame_seq: int = 0

    def open(self) -> None:
        """Validate path and open the OpenCV VideoCapture stream."""
        if not self._video_path.exists() or not self._video_path.is_file():
            raise FrameSourceError(f"Video file does not exist: '{self._video_path}'")

        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            cap.release()
            raise FrameSourceError(f"Failed to open video file: '{self._video_path}'")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0.0 or fps > 1000.0:
            fps = self._default_fps

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            cap.release()
            raise FrameSourceError(
                f"Invalid frame dimensions ({width}x{height}) in video '{self._video_path}'"
            )

        self._cap = cap
        self._fps = fps
        self._width = width
        self._height = height
        self._frame_seq = 0
        logger.info(
            "Opened video source '%s' (%dx%d @ %.2f FPS)",
            self._video_path,
            self._width,
            self._height,
            self._fps,
        )

    def read(self) -> Tuple[bool, Optional[AcquiredFrame], bool]:
        """Read next frame from video capture, distinguishing corrupted frames from EOF."""
        if self._cap is None or not self._cap.isOpened():
            return False, None, True

        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

        ret, frame = self._cap.read()
        if not ret or frame is None:
            if total_frames > 0 and current_pos < total_frames - 1:
                self._frame_seq += 1
                logger.warning("Corrupted or unreadable video frame at index %d", current_pos)
                return False, None, False
            return False, None, True

        if frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
            self._frame_seq += 1
            logger.warning("Malformed video frame shape at index %d", current_pos)
            return False, None, False

        self._frame_seq += 1
        timestamp = self._frame_seq / self._fps
        return (
            True,
            AcquiredFrame(
                frame_id=self._frame_seq,
                timestamp=timestamp,
                frame=frame,
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
                source_fps=self._fps,
            ),
            False,
        )

    def get_fps(self) -> float:
        """Return detected video FPS."""
        return self._fps

    def close(self) -> None:
        """Release VideoCapture safely."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


class IsaacSimRos2FrameSource(FrameSource):
    """
    NVIDIA Isaac Sim live camera integration via ROS 2 sensor_msgs/msg/Image subscription.

    Validates ROS 2 runtime availability, topic presence, image encoding conversion
    into BGR8, timestamp preservation, and bounded reconnection attempts.
    """

    def __init__(
        self,
        topic: str,
        expected_encoding: str,
        reconnect_enabled: bool,
        reconnect_attempts: int,
        default_fps: int = 30,
        timeout_sec: float = 2.5,
    ) -> None:
        self._topic = topic
        self._expected_encoding = expected_encoding.lower()
        self._reconnect_enabled = reconnect_enabled
        self._reconnect_attempts = reconnect_attempts
        self._fps = float(max(1, default_fps))
        self._timeout_sec = timeout_sec

        self._rclpy: Any = None
        self._node: Any = None
        self._sub: Any = None
        self._latest_msg: Any = None
        self._msg_event = threading.Event()
        self._frame_seq: int = 0
        self._initialized_rclpy_internally: bool = False

    def open(self) -> None:
        """Import rclpy and sensor_msgs, initialize node, and verify topic subscription."""
        try:
            import rclpy
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise FrameSourceError(
                "ROS 2 Python dependencies ('rclpy', 'sensor_msgs') are not installed in this "
                f"environment. Cannot start Isaac Sim live mode: {exc}"
            ) from exc

        self._rclpy = rclpy
        try:
            if not rclpy.ok():
                rclpy.init()
                self._initialized_rclpy_internally = True
            self._node = rclpy.create_node("amr_downfall_hazard_isaac_subscriber")
            self._sub = self._node.create_subscription(
                Image,
                self._topic,
                self._on_image_msg,
                10,
            )
        except Exception as exc:
            self.close()
            raise FrameSourceError(
                f"Failed to initialize ROS 2 subscriber on topic '{self._topic}': {exc}"
            ) from exc

        # Wait briefly to verify publisher presence on the configured topic
        deadline = time.monotonic() + self._timeout_sec
        while time.monotonic() < deadline:
            self._rclpy.spin_once(self._node, timeout_sec=0.1)
            if self._msg_event.is_set():
                logger.info("Connected to Isaac Sim ROS 2 image topic '%s'", self._topic)
                return

        if not self._reconnect_enabled:
            self.close()
            raise FrameSourceError(
                f"No ROS 2 messages received on Isaac Sim topic '{self._topic}' within {self._timeout_sec:.1f}s."
            )

        for attempt in range(1, self._reconnect_attempts + 1):
            logger.warning(
                "Waiting for Isaac Sim topic '%s' (reconnection attempt %d/%d)...",
                self._topic,
                attempt,
                self._reconnect_attempts,
            )
            self._rclpy.spin_once(self._node, timeout_sec=0.5)
            if self._msg_event.is_set():
                logger.info("Connected to Isaac Sim ROS 2 topic '%s' on attempt %d", self._topic, attempt)
                return

        self.close()
        raise FrameSourceError(
            f"Unable to receive frames from Isaac Sim ROS 2 topic '{self._topic}' "
            f"after {self._reconnect_attempts} reconnection attempts."
        )

    def _on_image_msg(self, msg: Any) -> None:
        """Callback storing the latest ROS 2 Image message."""
        self._latest_msg = msg
        self._msg_event.set()

    def read(self) -> Tuple[bool, Optional[AcquiredFrame], bool]:
        """Spin ROS 2 node once, decode image message into OpenCV BGR, and return AcquiredFrame."""
        if self._rclpy is None or self._node is None:
            return False, None, True

        self._msg_event.clear()
        self._rclpy.spin_once(self._node, timeout_sec=self._timeout_sec)

        if not self._msg_event.is_set() or self._latest_msg is None:
            if not self._reconnect_enabled:
                return False, None, True
            for attempt in range(1, self._reconnect_attempts + 1):
                logger.warning(
                    "Isaac Sim stream stalled on '%s'; reconnect attempt %d/%d",
                    self._topic,
                    attempt,
                    self._reconnect_attempts,
                )
                self._rclpy.spin_once(self._node, timeout_sec=0.5)
                if self._msg_event.is_set() and self._latest_msg is not None:
                    break
            else:
                return False, None, True

        msg = self._latest_msg
        self._latest_msg = None

        try:
            bgr_frame, timestamp = self.decode_ros_image(msg)
        except ValueError as exc:
            logger.error("Invalid ROS 2 Image message on '%s': %s", self._topic, exc)
            return False, None, False

        self._frame_seq += 1
        return (
            True,
            AcquiredFrame(
                frame_id=self._frame_seq,
                timestamp=timestamp,
                frame=bgr_frame,
                width=int(bgr_frame.shape[1]),
                height=int(bgr_frame.shape[0]),
                source_fps=self._fps,
            ),
            False,
        )

    @staticmethod
    def decode_ros_image(msg: Any) -> Tuple[np.ndarray, float]:
        """Convert a ROS 2 sensor_msgs/msg/Image message into an OpenCV BGR ndarray and timestamp."""
        width = int(getattr(msg, "width", 0))
        height = int(getattr(msg, "height", 0))
        encoding = str(getattr(msg, "encoding", "")).strip().lower()
        raw_data = getattr(msg, "data", None)

        if width <= 0 or height <= 0 or raw_data is None:
            raise ValueError(f"Invalid ROS Image dimensions ({width}x{height}) or empty data buffer.")

        if encoding not in ("bgr8", "rgb8", "rgba8", "bgra8", "mono8"):
            raise ValueError(f"Unsupported ROS Image encoding: '{encoding}'")

        channels = 1 if encoding == "mono8" else (4 if encoding in ("rgba8", "bgra8") else 3)
        expected_bytes = width * height * channels
        buf = np.frombuffer(bytes(raw_data), dtype=np.uint8)
        if buf.size < expected_bytes:
            raise ValueError(
                f"ROS Image buffer size ({buf.size}) is smaller than expected ({expected_bytes})."
            )

        img = buf[:expected_bytes].reshape((height, width, channels)) if channels > 1 else buf[:expected_bytes].reshape((height, width))

        if encoding == "bgr8":
            bgr = img.copy()
        elif encoding == "rgb8":
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif encoding == "rgba8":
            bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif encoding == "bgra8":
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None) if header is not None else None
        if stamp is not None and hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
            ts = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            if ts <= 0.0:
                ts = time.monotonic()
        else:
            ts = time.monotonic()

        return bgr, ts

    def get_fps(self) -> float:
        """Return configured camera FPS."""
        return self._fps

    def close(self) -> None:
        """Destroy ROS 2 subscription and node cleanly."""
        if self._node is not None:
            try:
                if self._sub is not None:
                    self._node.destroy_subscription(self._sub)
                self._node.destroy_node()
            except Exception:
                pass
            self._sub = None
            self._node = None
        if self._initialized_rclpy_internally and self._rclpy is not None:
            try:
                if self._rclpy.ok():
                    self._rclpy.shutdown()
            except Exception:
                pass
            self._initialized_rclpy_internally = False


class DownfallHazardApplication:
    """
    Central orchestrator for the Downfall Hazard Detection & AMR Risk Segregation system.

    Controls configuration, input source, YOLO detection, baseline profiling,
    motion gating, selective ByteTrack, landing verification, footprint extraction,
    Shapely corridor risk evaluation, incident evidence logging, circular frame buffering,
    and OpenCV visualization.
    """

    def __init__(
        self,
        config_path: str | Path = "config.properties",
        cli_overrides: Optional[Dict[str, str]] = None,
        injected_frame_source: Optional[FrameSource] = None,
        injected_detector: Optional[ObjectDetector] = None,
    ) -> None:
        self._config_path = Path(config_path)
        self._cli_overrides = cli_overrides or {}
        self._injected_frame_source = injected_frame_source
        self._injected_detector = injected_detector

        self.config: Optional[AppConfig] = None
        self.frame_source: Optional[FrameSource] = None
        self.detector: Optional[ObjectDetector] = None
        self.baseline_profiler: Optional[BaselineProfiler] = None
        self.motion_gate: Optional[MotionGate] = None
        self.track_manager: Optional[TrackManager] = None
        self.corridor: Optional[CorridorGeometry] = None
        self.risk_evaluator: Optional[RiskEvaluator] = None
        self.evidence_logger: Optional[EvidenceLogger] = None
        self.visualizer: Optional[Visualizer] = None

        self.frame_buffer: Deque[Tuple[float, np.ndarray]] = deque()
        self._buffer_max_frames: int = 300
        self._buffer_duration_sec: float = 10.0
        self.run_output_dir: Optional[Path] = None
        self.run_started_at: Optional[datetime] = None
        self._annotated_video_writer: Optional[cv2.VideoWriter] = None
        self._annotated_frame_count: int = 0

        self.active_tracks_registry: Dict[int, TrackedObject] = {}
        self.incident_registry: Dict[int, str] = {}
        self.risk_registry: Dict[int, RiskAssessmentResult] = {}
        self.telemetry = PerformanceTelemetry()
        self._is_running: bool = False
        self._is_paused: bool = False

    def initialize(self) -> None:
        """Execute full 14-step initialization sequence."""
        # 1 & 2. Load and validate configuration
        self.config = ConfigManager.load(self._config_path, overrides=self._cli_overrides)

        # 3. Initialize logging
        self._setup_logging()
        logger.info("Configuration loaded and validated from '%s'", self._config_path)

        # 4. Initialize input source
        self.initialize_input()

        # 5. Load YOLO detector
        self.initialize_detector()

        # 6, 7, 8. Initialize baseline profiler, motion gate, and ByteTrack
        self.initialize_tracking()

        # 9 & 10. Construct AMR corridor and initialize risk evaluator
        self.initialize_geometry()

        # 11. Create a unique output directory for this run and its evidence.
        self._prepare_run_output()

        # 12. Initialize evidence logger
        self.initialize_evidence()

        # 13. Initialize OpenCV visualization
        assert self.config is not None and self.corridor is not None
        self.visualizer = Visualizer(self.config.display, self.corridor)

        # 14. Initialize circular frame buffer (collections.deque)
        effective_fps = int(round(self.frame_source.get_fps())) if self.frame_source else self.config.storage.camera_fps
        effective_fps = max(1, effective_fps)
        self._buffer_duration_sec = self.config.storage.buffer_duration_sec
        self._buffer_max_frames = max(10, int(round(effective_fps * self._buffer_duration_sec)))
        self.frame_buffer = deque(maxlen=self._buffer_max_frames)

        # 15. Initialize active track and incident registries
        self.active_tracks_registry.clear()
        self.incident_registry.clear()
        self.risk_registry.clear()
        logger.info(
            "System initialization complete (buffer capacity=%d frames, %.1fs).",
            self._buffer_max_frames,
            self._buffer_duration_sec,
        )

    def _setup_logging(self) -> None:
        """Configure file and console logging under ./logs/."""
        log_dir = self._config_path.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "downfall_hazard.log"

        root_logger = logging.getLogger()
        if not root_logger.handlers:
            root_logger.setLevel(logging.INFO)
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
            )
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            root_logger.addHandler(fh)
            root_logger.addHandler(sh)

    def initialize_input(self) -> None:
        """Initialize the configured Video or Isaac Sim ROS 2 frame source."""
        assert self.config is not None
        if self._injected_frame_source is not None:
            self.frame_source = self._injected_frame_source
            self.frame_source.open()
            return

        if self.config.input.source == "video":
            self.frame_source = VideoFrameSource(
                video_path=self.config.input.video_path,
                default_fps=self.config.storage.camera_fps,
            )
        elif self.config.input.source == "isaac":
            self.frame_source = IsaacSimRos2FrameSource(
                topic=self.config.input.isaac_image_topic,
                expected_encoding=self.config.input.isaac_encoding,
                reconnect_enabled=self.config.input.reconnect_enabled,
                reconnect_attempts=self.config.input.reconnect_attempts,
                default_fps=self.config.storage.camera_fps,
            )
        else:
            raise ConfigurationError(f"Unsupported input source: {self.config.input.source}")

        self.frame_source.open()

    def initialize_detector(self) -> None:
        """Initialize the YOLO26 ObjectDetector."""
        assert self.config is not None
        if self._injected_detector is not None:
            self.detector = self._injected_detector
            return

        self.detector = ObjectDetector(
            config=self.config.model,
            geometry_epsilon=self.config.risk.geometry_epsilon,
        )

    def initialize_tracking(self) -> None:
        """Initialize BaselineProfiler, MotionGate, and selective TrackManager."""
        assert self.config is not None
        self.baseline_profiler = BaselineProfiler(self.config.baseline)
        self.motion_gate = MotionGate(
            config=self.config.motion,
            max_missed_frames=self.config.tracking.maximum_missed_frames,
        )
        fps = int(round(self.frame_source.get_fps())) if self.frame_source else self.config.storage.camera_fps
        self.track_manager = TrackManager(
            tracking_config=self.config.tracking,
            motion_config=self.config.motion,
            landing_config=self.config.landing,
            camera_fps=max(1, fps),
        )

    def initialize_geometry(self) -> None:
        """Construct the validated AMR corridor polygon and initialize RiskEvaluator."""
        assert self.config is not None
        self.corridor = ConfigManager.build_amr_corridor(self.config)
        self.risk_evaluator = RiskEvaluator(
            risk_config=self.config.risk,
            corridor=self.corridor,
            calibration_config=self.config.calibration,
        )

    def initialize_evidence(self) -> None:
        """Initialize the asynchronous EvidenceLogger."""
        assert self.config is not None and self.corridor is not None
        assert self.run_output_dir is not None
        self.evidence_logger = EvidenceLogger(
            storage_config=replace(
                self.config.storage,
                evidence_base_dir=str(self.run_output_dir / "incidents"),
            ),
            corridor=self.corridor,
            amr_config=self.config.amr,
        )

    def _prepare_run_output(self) -> None:
        """Create a unique evidence directory so consecutive runs never overwrite each other."""
        assert self.config is not None
        timestamp = datetime.now(timezone.utc)
        run_name = f"run_{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}"
        self.run_output_dir = Path(self.config.storage.evidence_base_dir) / run_name
        (self.run_output_dir / "incidents").mkdir(parents=True, exist_ok=False)
        self.run_started_at = timestamp

    def update_frame_buffer(self, frame: np.ndarray, timestamp: float) -> None:
        """
        Store an independent copy of frame in the circular buffer and prune frames
        older than buffer_duration_sec.
        """
        self.frame_buffer.append((timestamp, frame.copy()))
        cutoff = timestamp - self._buffer_duration_sec - 1e-6
        while len(self.frame_buffer) > 1 and self.frame_buffer[0][0] < cutoff:
            self.frame_buffer.popleft()

    def perform_detection(
        self, frame: np.ndarray, frame_id: int, timestamp: float
    ) -> DetectionFrameOutput:
        """Run YOLO26 inference on the current frame."""
        assert self.detector is not None
        return self.detector.detect(frame, frame_id=frame_id, timestamp=timestamp)

    def update_baseline(
        self, detections: List[Detection], timestamp: float
    ) -> BaselineUpdateResult:
        """Update baseline stationary debris registry and filter pre-existing debris."""
        assert self.baseline_profiler is not None
        return self.baseline_profiler.update(detections, timestamp=timestamp)

    def perform_motion_association(
        self, non_debris_detections: List[Detection], timestamp: float
    ) -> MotionGateOutput:
        """Execute lightweight motion pre-association and downward velocity gating."""
        assert self.motion_gate is not None and self.track_manager is not None
        active_bboxes = self.track_manager.get_active_track_bboxes()
        return self.motion_gate.evaluate(
            detections=non_debris_detections,
            timestamp=timestamp,
            active_track_bboxes=active_bboxes,
        )

    def update_tracks(
        self,
        motion_output: MotionGateOutput,
        frame_id: int,
        timestamp: float,
    ) -> TrackingFrameOutput:
        """Update selective ByteTrack with qualifying descending candidates."""
        assert self.track_manager is not None
        output = self.track_manager.update(
            qualifying_detections=motion_output.qualifying_detections,
            all_non_debris_detections=motion_output.evaluated_detections,
            frame_id=frame_id,
            timestamp=timestamp,
        )
        self.active_tracks_registry = {t.track_id: t for t in output.active_tracks}
        for exp_id in output.expired_track_ids:
            self.active_tracks_registry.pop(exp_id, None)
        return output

    def verify_landing(self, frame_id: int, timestamp: float) -> List[TrackedObject]:
        """Verify landing stability across all active falling tracks."""
        assert self.track_manager is not None
        return self.track_manager.verify_landing(frame_id=frame_id, timestamp=timestamp)

    def extract_footprint(self, track: TrackedObject) -> Optional[Polygon]:
        """Extract or validate the final resting footprint polygon for a landed track."""
        assert self.config is not None
        if (
            track.footprint is not None
            and isinstance(track.footprint, Polygon)
            and track.footprint.is_valid
            and not track.footprint.is_empty
            and track.footprint.area > self.config.risk.geometry_epsilon
        ):
            return track.footprint

        fallback, _, _ = ObjectDetector.extract_footprint_from_mask_or_bbox(
            track.mask_contour, track.bbox, epsilon=self.config.risk.geometry_epsilon
        )
        if fallback is not None:
            track.footprint = fallback
        return fallback

    def evaluate_risk(
        self,
        footprint: Optional[Polygon],
        corridor_contact_history: bool = False,
    ) -> RiskAssessmentResult:
        """Compute Shapely intersection with the AMR corridor and classify hazard risk."""
        assert self.risk_evaluator is not None
        return self.risk_evaluator.evaluate(
            footprint,
            corridor_contact_history=corridor_contact_history,
        )

    def generate_incident(
        self,
        track: TrackedObject,
        risk_result: RiskAssessmentResult,
        frame_id: int,
        current_frame: np.ndarray,
    ) -> Optional[str]:
        """
        Central decision point: only main.py decides when a confirmed landing
        becomes a recorded incident.
        """
        assert self.evidence_logger is not None and self.baseline_profiler is not None

        # Never create incidents while initial baseline observation window is still running
        if not self.baseline_profiler.is_baseline_complete:
            logger.info(
                "Skipping incident for track %d because baseline observation is still active.",
                track.track_id,
            )
            return None

        # Prevent duplicate incidents for the same track
        if track.incident_generated or track.track_id in self.incident_registry:
            return None
        if self.evidence_logger.has_incident_for_track(track.track_id):
            return None

        pre_frames = [frm for _, frm in self.frame_buffer]
        event_id = self.evidence_logger.start_incident(
            track=track,
            risk_result=risk_result,
            frame_id=frame_id,
            current_frame=current_frame,
            pre_event_frames=pre_frames,
            is_preexisting_debris=False,
        )
        if event_id is not None:
            track.incident_generated = True
            track.lifecycle_state = LifecycleState.INCIDENT_RECORDED
            self.incident_registry[track.track_id] = event_id
            self.telemetry.incidents_generated = len(self.incident_registry)
            logger.info(
                "Generated incident '%s' for track_id=%d (%s) with risk=%s (overlap=%.2f%%)",
                event_id,
                track.track_id,
                track.class_name,
                risk_result.risk_level.value,
                risk_result.overlap_percentage,
            )
        else:
            track.lifecycle_state = LifecycleState.PROCESSING_FAILED

        return event_id

    def render_frame(
        self,
        frame: np.ndarray,
        baseline_result: BaselineUpdateResult,
        motion_output: MotionGateOutput,
        detections: List[Detection],
    ) -> np.ndarray:
        """Render OpenCV monitoring overlays onto the current frame."""
        assert self.visualizer is not None
        return self.visualizer.render(
            frame=frame,
            detections=detections,
            active_tracks=list(self.active_tracks_registry.values()),
            baseline_debris=baseline_result.active_debris,
            evaluated_detections=motion_output.evaluated_detections,
            risk_by_track_id=self.risk_registry,
            fps=self.telemetry.effective_fps,
            incident_count=len(self.incident_registry),
            baseline_complete=baseline_result.baseline_complete,
        )

    def process_frame(self, acquired: AcquiredFrame) -> FrameProcessingResult:
        """Execute the complete downfall detection and risk segregation pipeline for one frame."""
        assert (
            self.config is not None
            and self.evidence_logger is not None
        )
        t_start = time.perf_counter()

        frame = acquired.frame
        frame_id = acquired.frame_id
        timestamp = acquired.timestamp

        # 1. Update circular pre-event frame buffer
        self.update_frame_buffer(frame, timestamp)

        # 2. Feed frame to any pending post-event video clip recorders
        self.evidence_logger.ingest_post_event_frame(frame)

        # 3. YOLO26 Inference
        det_output = self.perform_detection(frame, frame_id=frame_id, timestamp=timestamp)
        self.telemetry.yolo_latency_ms = det_output.inference_latency_ms

        # 4. Baseline Registration & Pre-existing Debris Filtering
        baseline_result = self.update_baseline(det_output.detections, timestamp=timestamp)

        # 5. Lightweight Motion Pre-Association & Downfall Gating
        motion_output = self.perform_motion_association(
            baseline_result.non_debris_detections, timestamp=timestamp
        )

        # 6. Selective ByteTrack & Persistent Track Management
        track_output = self.update_tracks(
            motion_output=motion_output,
            frame_id=frame_id,
            timestamp=timestamp,
        )
        self.telemetry.tracking_latency_ms = track_output.tracking_latency_ms

        for track in self.active_tracks_registry.values():
            current_risk = self.evaluate_risk(self.extract_footprint(track))
            if current_risk.geometry_valid and (
                current_risk.intersection_area > self.config.risk.geometry_epsilon
                or current_risk.touches_boundary
            ):
                track.corridor_contact = True

        # 7. Landing Verification
        newly_landed = self.verify_landing(frame_id=frame_id, timestamp=timestamp)

        # 8. Footprint Extraction, AMR Corridor Intersection, Risk Classification & Incident Generation
        geom_start = time.perf_counter()
        newly_landed_ids: List[int] = []
        triggered_incidents: List[str] = []

        for landed_track in newly_landed:
            newly_landed_ids.append(landed_track.track_id)
            footprint = self.extract_footprint(landed_track)
            risk_result = self.evaluate_risk(
                footprint,
                corridor_contact_history=landed_track.corridor_contact,
            )
            landed_track.risk_level = risk_result.risk_level.value
            landed_track.overlap_percentage = risk_result.overlap_percentage
            landed_track.lifecycle_state = LifecycleState.RISK_ASSESSED
            self.risk_registry[landed_track.track_id] = risk_result

            event_id = self.generate_incident(
                track=landed_track,
                risk_result=risk_result,
                frame_id=frame_id,
                current_frame=frame,
            )
            if event_id is not None:
                triggered_incidents.append(event_id)

        self.telemetry.geometry_latency_ms = (time.perf_counter() - geom_start) * 1000.0

        # 9. OpenCV Visualization Rendering
        rendered = self.render_frame(
            frame,
            baseline_result,
            motion_output,
            detections=det_output.detections,
        )

        total_ms = (time.perf_counter() - t_start) * 1000.0
        self.telemetry.total_frame_latency_ms = total_ms
        self.telemetry.processed_frames += 1
        inst_fps = 1000.0 / max(0.1, total_ms)
        if self.telemetry.effective_fps <= 0.0:
            self.telemetry.effective_fps = inst_fps
        else:
            self.telemetry.effective_fps = 0.85 * self.telemetry.effective_fps + 0.15 * inst_fps
        self.telemetry.active_tracks = len(self.active_tracks_registry)
        self.telemetry.pending_evidence_jobs = self.evidence_logger.pending_job_count

        return FrameProcessingResult(
            frame_id=frame_id,
            timestamp=timestamp,
            detections_count=len(det_output.detections),
            non_debris_count=len(baseline_result.non_debris_detections),
            qualifying_motion_count=len(motion_output.qualifying_detections),
            active_tracks_count=len(self.active_tracks_registry),
            newly_landed_track_ids=newly_landed_ids,
            triggered_incident_ids=triggered_incidents,
            risk_results=dict(self.risk_registry),
            rendered_frame=rendered,
            telemetry=self.telemetry,
        )

    def run(self) -> int:
        """Run the central frame acquisition and processing loop until completion or user quit."""
        self.initialize()
        assert (
            self.config is not None
            and self.frame_source is not None
            and self.visualizer is not None
        )

        self._is_running = True
        if self.config.performance.frame_skip > 0:
            logger.warning(
                "Frame skipping is disabled so the annotated video includes every input frame."
            )
        raw_frame_counter = 0
        min_frame_interval = (
            1.0 / self.config.performance.max_processing_fps
            if self.config.performance.max_processing_fps > 0.0
            else 0.0
        )
        last_Result_rendered: Optional[np.ndarray] = None

        try:
            while self._is_running:
                if self._is_paused and last_Result_rendered is not None:
                    cmd = self.visualizer.show_and_poll_key(last_Result_rendered, delay_ms=30)
                    if cmd.quit_requested:
                        logger.info("Quit requested by user during pause.")
                        break
                    if cmd.toggle_pause:
                        self._is_paused = False
                        logger.info("Resumed processing.")
                    if cmd.save_screenshot:
                        saved = self.visualizer.save_diagnostic_screenshot(
                            last_Result_rendered,
                            self.config.storage.evidence_base_dir,
                            raw_frame_counter,
                        )
                        logger.info("Saved diagnostic screenshot: %s", saved)
                    continue

                loop_start = time.perf_counter()
                ok, acquired, is_eof = self.frame_source.read()

                if is_eof:
                    logger.info("End of input stream reached after %d frames.", raw_frame_counter)
                    break

                if not ok or acquired is None:
                    self.telemetry.dropped_frames += 1
                    continue

                raw_frame_counter += 1
                result = self.process_frame(acquired)
                last_Result_rendered = result.rendered_frame
                self._write_annotated_frame(result.rendered_frame, acquired.source_fps)

                cmd = self.visualizer.show_and_poll_key(result.rendered_frame, delay_ms=1)
                if cmd.quit_requested:
                    logger.info("Quit requested by user ('q').")
                    break
                if cmd.toggle_pause:
                    self._is_paused = not self._is_paused
                    logger.info("Paused state toggled: %s", self._is_paused)
                if cmd.save_screenshot:
                    saved = self.visualizer.save_diagnostic_screenshot(
                        result.rendered_frame,
                        self.config.storage.evidence_base_dir,
                        result.frame_id,
                    )
                    logger.info("Saved diagnostic screenshot: %s", saved)

                if min_frame_interval > 0.0:
                    elapsed = time.perf_counter() - loop_start
                    if elapsed < min_frame_interval:
                        time.sleep(min_frame_interval - elapsed)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received; shutting down gracefully.")
        finally:
            self.shutdown()

        return 0

    def _write_annotated_frame(self, frame: np.ndarray, source_fps: float) -> None:
        """Append one rendered frame to this run's complete annotated video."""
        assert self.config is not None and self.run_output_dir is not None
        if self._annotated_video_writer is None:
            codec = self.config.storage.video_codec
            if len(codec) != 4:
                raise ValueError("storage.video_codec must contain exactly four characters.")
            fps = source_fps if np.isfinite(source_fps) and source_fps > 0.0 else float(
                self.config.storage.camera_fps
            )
            video_path = self.run_output_dir / "annotated_video.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*codec),
                fps,
                (int(frame.shape[1]), int(frame.shape[0])),
            )
            if not writer.isOpened():
                writer.release()
                raise RuntimeError(f"Could not create annotated video: {video_path}")
            self._annotated_video_writer = writer

        self._annotated_video_writer.write(frame)
        self._annotated_frame_count += 1

    def _write_run_manifest(
        self, completed_evidence: List[CompletedIncidentRecord]
    ) -> None:
        """Persist a summary of this run and its generated evidence files."""
        if self.run_output_dir is None or self.config is None:
            return
        video_path = self.run_output_dir / "annotated_video.mp4"
        manifest = {
            "run_id": self.run_output_dir.name,
            "started_at_utc": self.run_started_at.isoformat()
            if self.run_started_at is not None
            else None,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_source": self.config.input.source,
            "input_video": self.config.input.video_path
            if self.config.input.source == "video"
            else None,
            "processed_frames": self.telemetry.processed_frames,
            "dropped_frames": self.telemetry.dropped_frames,
            "incidents_generated": len(self.incident_registry),
            "annotated_video": video_path.name if video_path.exists() else None,
            "annotated_video_frames": self._annotated_frame_count,
            "evidence_records": [
                {
                    "event_id": record.event_id,
                    "track_id": record.track_id,
                    "success": record.success,
                    "incident_directory": str(
                        record.incident_dir.relative_to(self.run_output_dir)
                    ),
                    "error": record.error_message,
                }
                for record in completed_evidence
            ],
        }
        manifest_path = self.run_output_dir / "run_metadata.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("Run outputs saved to '%s'", self.run_output_dir)

    def shutdown(self) -> List[CompletedIncidentRecord]:
        """Release camera, flush pending evidence recordings, and close OpenCV windows."""
        self._is_running = False
        completed_evidence: List[CompletedIncidentRecord] = []

        if self.frame_source is not None:
            try:
                self.frame_source.close()
            except Exception as exc:
                logger.error("Error closing frame source: %s", exc)

        if self.evidence_logger is not None:
            try:
                completed_evidence = self.evidence_logger.flush_and_shutdown()
            except Exception as exc:
                logger.error("Error flushing evidence logger: %s", exc)

        if self._annotated_video_writer is not None:
            try:
                self._annotated_video_writer.release()
            except Exception as exc:
                logger.error("Error closing annotated video writer: %s", exc)
            self._annotated_video_writer = None

        if self.visualizer is not None:
            try:
                self.visualizer.close()
            except Exception as exc:
                logger.error("Error closing visualizer: %s", exc)

        try:
            self._write_run_manifest(completed_evidence)
        except Exception as exc:
            logger.error("Error writing run metadata: %s", exc)

        logger.info(
            "Shutdown complete. Processed=%d, Dropped=%d, Incidents=%d, VerifiedEvidence=%d",
            self.telemetry.processed_frames,
            self.telemetry.dropped_frames,
            len(self.incident_registry),
            sum(1 for r in completed_evidence if r.success),
        )
        return completed_evidence


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser supporting explicit overrides."""
    parser = argparse.ArgumentParser(
        description="Real-Time Downfall Hazard Detection & AMR Risk Segregation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.properties",
        help="Path to config.properties file (default: config.properties)",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["video", "isaac"],
        default=None,
        help="Override input.source ('video' or 'isaac')",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Override input.video_path for video mode",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override model.path for YOLO26 checkpoint",
    )
    parser.add_argument(
        "--display",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Override display.enabled ('true' or 'false')",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    overrides: Dict[str, str] = {}
    if args.source is not None:
        overrides["input.source"] = args.source
    if args.video_path is not None:
        overrides["input.video_path"] = args.video_path
    if args.model_path is not None:
        overrides["model.path"] = args.model_path
    if args.display is not None:
        overrides["display.enabled"] = args.display

    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        candidate = Path(__file__).resolve().parent / args.config
        if candidate.exists():
            config_path = candidate

    try:
        app = DownfallHazardApplication(config_path=config_path, cli_overrides=overrides)
        return app.run()
    except (ConfigurationError, FrameSourceError, DetectorError) as exc:
        logger.error("Application startup or runtime error: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unhandled fatal error: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
