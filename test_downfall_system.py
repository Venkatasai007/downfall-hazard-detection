"""
Comprehensive executable test suite for Downfall Hazard Detection & AMR Risk Segregation.

Covers all 27 required verification areas:
1. Configuration validation
2. Corridor polygon construction
3. Baseline registration
4. Existing debris movement
5. Downward velocity
6. Upward movement rejection
7. Horizontal movement rejection
8. Selective ByteTrack activation
9. Persistent tracking
10. Temporary occlusion
11. Landing confirmation
12. Bounce and rolling
13. Duplicate incident prevention
14. Segmentation footprint extraction
15. Bounding-box fallback
16. HIGH classification
17. MEDIUM classification
18. LOW classification
19. Boundary contact
20. UNCERTAIN geometry
21. Circular frame buffer
22. Pre-event recording
23. Post-event recording
24. Snapshot creation
25. MP4 creation
26. JSON validation
27. Synthetic-video end-to-end integration

Compatible with both `pytest` and `python3 -m unittest`.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import unittest

import cv2
import numpy as np
from shapely.geometry import Polygon, box

from baseline_profiler import BaselineProfiler
from config_manager import (
    AmrConfig,
    AppConfig,
    CalibrationConfig,
    ConfigManager,
    ConfigurationError,
    RiskConfig,
)
from detector import Detection, ObjectDetector
from evidence_logger import EvidenceLogger
from main import DownfallHazardApplication, FrameSourceError, IsaacSimRos2FrameSource
from motion_gate import LifecycleState, MotionGate
from risk_evaluator import RiskEvaluator, RiskLevel
from track_manager import TrackManager


class FakeTensor:
    """Lightweight tensor mock mimicking Ultralytics/PyTorch .cpu().numpy() interface."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data, dtype=np.float64)

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._data

    def __len__(self) -> int:
        return len(self._data)


class FakeBoxes:
    """Mock Ultralytics Boxes container."""

    def __init__(
        self,
        xyxy: List[Tuple[float, float, float, float]],
        conf: List[float],
        cls: List[int],
    ) -> None:
        self.xyxy = FakeTensor(np.array(xyxy, dtype=np.float64))
        self.conf = FakeTensor(np.array(conf, dtype=np.float64))
        self.cls = FakeTensor(np.array(cls, dtype=np.float64))

    def __len__(self) -> int:
        return len(self.conf)


class FakeMasks:
    """Mock Ultralytics Masks container."""

    def __init__(self, polygons: List[np.ndarray]) -> None:
        self.xy = polygons


class FakeResult:
    """Mock Ultralytics Result object."""

    def __init__(
        self,
        boxes: Optional[FakeBoxes],
        masks: Optional[FakeMasks] = None,
        names: Optional[Dict[int, str]] = None,
    ) -> None:
        self.boxes = boxes
        self.masks = masks
        self.names = names or {0: "box", 1: "pallet"}


class ScriptedYoloModel:
    """Deterministic scripted YOLO26 mock model for unit and synthetic video tests."""

    def __init__(
        self,
        frame_plan: Dict[int, FakeResult],
        task: str = "segment",
    ) -> None:
        self.task = task
        self.names = {0: "box", 1: "crate"}
        self._frame_plan = frame_plan
        self._call_idx = 0

    def predict(self, source: Any, **kwargs: Any) -> List[FakeResult]:
        self._call_idx += 1
        res = self._frame_plan.get(self._call_idx)
        if res is None:
            return [FakeResult(boxes=None, masks=None, names=self.names)]
        return [res]


def make_detection(
    det_id: int,
    bbox: Tuple[float, float, float, float],
    class_id: int = 0,
    class_name: str = "box",
    confidence: float = 0.92,
) -> Detection:
    """Helper to build a validated Detection instance."""
    x1, y1, x2, y2 = bbox
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    poly = box(x1, y1, x2, y2)
    return Detection(
        detection_id=det_id,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        centroid=(xc, yc),
        footprint=poly,
        has_segmentation_mask=False,
        mask_contour=None,
    )


class TestDownfallHazardSystem(unittest.TestCase):
    """Complete unit and integration test suite covering all 27 required scenarios."""

    def setUp(self) -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.config_path = self.base_dir / "config.properties"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="downfall_test_"))
        self.default_config = ConfigManager.load(
            self.config_path,
            overrides={"storage.evidence_base_dir": str(self.temp_dir / "evidence")},
        )

    def tearDown(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. Configuration validation
    def test_01_configuration_validation(self) -> None:
        cfg = self.default_config
        self.assertEqual(cfg.input.source, "video")
        self.assertAlmostEqual(cfg.model.conf_threshold, 0.35)
        self.assertEqual(cfg.corridor.yellow_line_p1, (320.0, 100.0))
        self.assertEqual(cfg.corridor.yellow_line_p2, (320.0, 900.0))
        self.assertAlmostEqual(cfg.risk.high_overlap_threshold, 10.0)

        # Invalid source
        with self.assertRaises(ConfigurationError):
            ConfigManager.load(self.config_path, overrides={"input.source": "invalid_cam"})

        # Out-of-range confidence
        with self.assertRaises(ConfigurationError):
            ConfigManager.load(self.config_path, overrides={"model.conf_threshold": "1.5"})

        # Low threshold > High threshold
        with self.assertRaises(ConfigurationError):
            ConfigManager.load(
                self.config_path,
                overrides={
                    "tracking.byte_track_low_threshold": "0.80",
                    "tracking.byte_track_high_threshold": "0.50",
                },
            )

        # Metric mode without homography
        with self.assertRaises(ConfigurationError):
            ConfigManager.load(
                self.config_path,
                overrides={
                    "corridor.geometry_mode": "metric",
                    "calibration.homography_enabled": "false",
                },
            )

        # Missing required key
        bad_props = self.temp_dir / "bad.properties"
        bad_props.write_text("input.source=video\n", encoding="utf-8")
        with self.assertRaises(ConfigurationError):
            ConfigManager.load(bad_props)

    # 2. Corridor polygon construction
    def test_02_corridor_polygon_construction(self) -> None:
        corridor = ConfigManager.build_amr_corridor(self.default_config)
        # P1=(320,100), P2=(320,900) -> L=800, W = 40 + 20 = 60 -> total width = 120 -> area = 800 * 120 = 96000
        self.assertTrue(corridor.polygon.is_valid)
        self.assertAlmostEqual(corridor.half_width_working, 60.0)
        self.assertAlmostEqual(corridor.polygon.area, 96000.0, places=3)
        self.assertAlmostEqual(corridor.normal_vector[0], -1.0, places=5)
        self.assertAlmostEqual(corridor.normal_vector[1], 0.0, places=5)

        # Reject zero-length centerline
        with self.assertRaises(ConfigurationError):
            ConfigManager.load(
                self.config_path,
                overrides={
                    "corridor.yellow_line_p1": "320,100",
                    "corridor.yellow_line_p2": "320,100",
                },
            )

        # Metric mode with valid homography scaling (100 px = 1 meter -> 0.01 scale)
        metric_cfg = ConfigManager.load(
            self.config_path,
            overrides={
                "corridor.geometry_mode": "metric",
                "calibration.homography_enabled": "true",
                "calibration.homography_matrix": "0.01,0,0, 0,0.01,0, 0,0,1",
            },
        )
        metric_corridor = ConfigManager.build_amr_corridor(metric_cfg)
        self.assertTrue(metric_corridor.polygon.is_valid)
        # P1=(3.2, 1.0), P2=(3.2, 9.0) -> L=8.0m, W = 0.50 + 0.25 = 0.75m -> total width = 1.5m -> area = 12.0 m^2
        self.assertAlmostEqual(metric_corridor.polygon.area, 12.0, places=4)

    # 3. Baseline registration
    def test_03_baseline_registration(self) -> None:
        profiler = BaselineProfiler(self.default_config.baseline)
        static_det = make_detection(1, (100.0, 200.0, 150.0, 250.0))

        # Single frame must NOT mark baseline complete or register immediately
        res_0 = profiler.update([static_det], timestamp=0.0)
        self.assertFalse(res_0.baseline_complete)

        # Observe across observation_duration_sec (3.0s) with >= 5 observations
        for step in range(1, 7):
            res = profiler.update([static_det], timestamp=step * 0.6)

        self.assertTrue(res.baseline_complete)
        debris_list = profiler.get_registered_debris()
        self.assertEqual(len(debris_list), 1)
        self.assertEqual(debris_list[0].label, "UNKNOWN_DEBRIS")
        self.assertTrue(profiler.is_preexisting_debris(static_det, threshold=0.70))
        self.assertEqual(len(res.non_debris_detections), 0)

    # 4. Existing debris movement
    def test_04_existing_debris_movement(self) -> None:
        profiler = BaselineProfiler(self.default_config.baseline)
        static_det = make_detection(1, (100.0, 200.0, 150.0, 250.0))
        for step in range(7):
            profiler.update([static_det], timestamp=step * 0.6)

        self.assertTrue(profiler.is_preexisting_debris(static_det))

        # Now displace the baseline object significantly downward
        displaced_det = make_detection(1, (100.0, 240.0, 150.0, 290.0))
        res_moved = profiler.update([displaced_det], timestamp=4.5)
        self.assertEqual(len(res_moved.non_debris_detections), 1)
        self.assertFalse(profiler.is_preexisting_debris(displaced_det))

    # 5. Downward velocity
    def test_05_downward_velocity_qualification(self) -> None:
        gate = MotionGate(self.default_config.motion)
        # Frame 1 at y_center=120, Frame 2 at y_center=150 after 0.1s -> vy = 300 px/s >= 60 px/s, disp = 30 px >= 5 px
        d1 = make_detection(1, (300.0, 100.0, 340.0, 140.0))
        d2 = make_detection(1, (300.0, 130.0, 340.0, 170.0))

        out1 = gate.evaluate([d1], timestamp=0.0)
        self.assertEqual(len(out1.qualifying_detections), 0)

        out2 = gate.evaluate([d2], timestamp=0.1)
        self.assertEqual(len(out2.qualifying_detections), 1)
        self.assertAlmostEqual(out2.qualifying_detections[0].instantaneous_vy, 300.0, places=2)
        self.assertEqual(
            out2.qualifying_detections[0].state, LifecycleState.DESCENDING_CANDIDATE
        )

    # 6. Upward movement rejection
    def test_06_upward_movement_rejection(self) -> None:
        gate = MotionGate(self.default_config.motion)
        d1 = make_detection(1, (300.0, 300.0, 340.0, 340.0))
        d2 = make_detection(1, (300.0, 260.0, 340.0, 300.0))  # Moved up by 40 px in 0.1s -> vy = -400 px/s

        gate.evaluate([d1], timestamp=0.0)
        out2 = gate.evaluate([d2], timestamp=0.1)
        self.assertEqual(len(out2.qualifying_detections), 0)
        self.assertEqual(out2.evaluated_detections[0].state, LifecycleState.REJECTED)
        self.assertEqual(out2.evaluated_detections[0].rejection_reason, "upward_motion")

    # 7. Horizontal movement rejection
    def test_07_horizontal_movement_rejection(self) -> None:
        gate = MotionGate(self.default_config.motion)
        d1 = make_detection(1, (100.0, 300.0, 140.0, 340.0))
        d2 = make_detection(1, (150.0, 301.0, 190.0, 341.0))  # vx = 500 px/s, vy = 10 px/s

        gate.evaluate([d1], timestamp=0.0)
        out2 = gate.evaluate([d2], timestamp=0.1)
        self.assertEqual(len(out2.qualifying_detections), 0)
        self.assertEqual(out2.evaluated_detections[0].state, LifecycleState.REJECTED)
        self.assertEqual(out2.evaluated_detections[0].rejection_reason, "horizontal_motion")

    # 8. Selective ByteTrack activation
    def test_08_selective_bytetrack_activation(self) -> None:
        gate = MotionGate(self.default_config.motion)
        tm = TrackManager(
            self.default_config.tracking,
            self.default_config.motion,
            self.default_config.landing,
        )

        # Stationary object + upward object should not create any track in TrackManager
        s1 = make_detection(1, (50.0, 50.0, 90.0, 90.0))
        u1 = make_detection(2, (200.0, 300.0, 240.0, 340.0))
        u2 = make_detection(2, (200.0, 260.0, 240.0, 300.0))

        m1 = gate.evaluate([s1, u1], timestamp=0.0)
        t_out1 = tm.update(m1.qualifying_detections, m1.evaluated_detections, frame_id=1, timestamp=0.0)
        self.assertEqual(len(t_out1.active_tracks), 0)

        m2 = gate.evaluate([s1, u2], timestamp=0.1)
        t_out2 = tm.update(m2.qualifying_detections, m2.evaluated_detections, frame_id=2, timestamp=0.1)
        self.assertEqual(len(t_out2.active_tracks), 0)

        # Now add a falling object
        f1 = make_detection(3, (300.0, 100.0, 340.0, 140.0))
        f2 = make_detection(3, (300.0, 130.0, 340.0, 170.0))
        gate.evaluate([f1], timestamp=0.2)
        m4 = gate.evaluate([f2], timestamp=0.3)
        t_out4 = tm.update(m4.qualifying_detections, m4.evaluated_detections, frame_id=4, timestamp=0.3)
        self.assertEqual(len(t_out4.active_tracks), 1)

    # 9. Persistent tracking & 10. Temporary occlusion & 11. Landing confirmation
    def test_09_10_11_persistent_tracking_occlusion_and_landing(self) -> None:
        gate = MotionGate(self.default_config.motion)
        tm = TrackManager(
            self.default_config.tracking,
            self.default_config.motion,
            self.default_config.landing,
        )

        dt = 1.0 / 30.0
        timestamp = 0.0
        frame_id = 0

        # Fall for 6 frames (y increases by 12 px per frame -> vy = 360 px/s)
        y = 100.0
        track_id_initial: Optional[int] = None
        for i in range(6):
            frame_id += 1
            timestamp = frame_id * dt
            y += 12.0
            det = make_detection(1, (300.0, y, 340.0, y + 40.0))
            m_out = gate.evaluate([det], timestamp, tm.get_active_track_bboxes())
            t_out = tm.update(m_out.qualifying_detections, m_out.evaluated_detections, frame_id, timestamp)
            if t_out.active_tracks:
                if track_id_initial is None:
                    track_id_initial = t_out.active_tracks[0].track_id
                else:
                    self.assertEqual(t_out.active_tracks[0].track_id, track_id_initial)

        self.assertIsNotNone(track_id_initial)

        # Rest for 5 frames
        for _ in range(5):
            frame_id += 1
            timestamp = frame_id * dt
            det = make_detection(1, (300.0, y, 340.0, y + 40.0))
            m_out = gate.evaluate([det], timestamp, tm.get_active_track_bboxes())
            tm.update(m_out.qualifying_detections, m_out.evaluated_detections, frame_id, timestamp)
            landed = tm.verify_landing(frame_id, timestamp)
            self.assertEqual(len(landed), 0)

        # Temporary occlusion for 2 frames: missed_frames increments, landing_counter must NOT increment
        counter_before_occlusion = tm.get_active_tracks()[0].landing_counter
        for _ in range(2):
            frame_id += 1
            timestamp = frame_id * dt
            m_out = gate.evaluate([], timestamp, tm.get_active_track_bboxes())
            tm.update(m_out.qualifying_detections, m_out.evaluated_detections, frame_id, timestamp)
            landed = tm.verify_landing(frame_id, timestamp)
            self.assertEqual(len(landed), 0)
            self.assertEqual(tm.get_active_tracks()[0].landing_counter, counter_before_occlusion)

        # Resume detections at resting position until landing_counter reaches 15
        confirmed_landed = []
        for _ in range(20):
            frame_id += 1
            timestamp = frame_id * dt
            det = make_detection(1, (300.0, y, 340.0, y + 40.0))
            m_out = gate.evaluate([det], timestamp, tm.get_active_track_bboxes())
            tm.update(m_out.qualifying_detections, m_out.evaluated_detections, frame_id, timestamp)
            landed = tm.verify_landing(frame_id, timestamp)
            if landed:
                confirmed_landed.extend(landed)
                break

        self.assertEqual(len(confirmed_landed), 1)
        self.assertEqual(confirmed_landed[0].track_id, track_id_initial)
        self.assertEqual(confirmed_landed[0].lifecycle_state, LifecycleState.LANDED)

    # 12. Bounce and rolling
    def test_12_bounce_and_rolling_resets_landing_counter(self) -> None:
        gate = MotionGate(self.default_config.motion)
        tm = TrackManager(
            self.default_config.tracking,
            self.default_config.motion,
            self.default_config.landing,
        )

        dt = 1.0 / 30.0
        frame_id = 0
        y = 100.0

        # Fall for 5 frames
        for _ in range(5):
            frame_id += 1
            y += 15.0
            det = make_detection(1, (300.0, y, 340.0, y + 40.0))
            m_out = gate.evaluate([det], frame_id * dt, tm.get_active_track_bboxes())
            tm.update(m_out.qualifying_detections, m_out.evaluated_detections, frame_id, frame_id * dt)
            tm.verify_landing(frame_id, frame_id * dt)

        # Pause for 8 frames (< 15 required rest frames)
        for _ in range(8):
            frame_id += 1
            det = make_detection(1, (300.0, y, 340.0, y + 40.0))
            m_out = gate.evaluate([det], frame_id * dt, tm.get_active_track_bboxes())
            tm.update(m_out.qualifying_detections, m_out.evaluated_detections, frame_id, frame_id * dt)
            landed = tm.verify_landing(frame_id, frame_id * dt)
            self.assertEqual(len(landed), 0)

        self.assertGreater(tm.get_active_tracks()[0].landing_counter, 0)

        # Bounce / roll: sudden displacement of 25 px (> 5.0 px stability threshold and high velocity)
        frame_id += 1
        y += 25.0
        det_bounce = make_detection(1, (315.0, y, 355.0, y + 40.0))
        m_out = gate.evaluate([det_bounce], frame_id * dt, tm.get_active_track_bboxes())
        tm.update(m_out.qualifying_detections, m_out.evaluated_detections, frame_id, frame_id * dt)
        landed = tm.verify_landing(frame_id, frame_id * dt)
        self.assertEqual(len(landed), 0)
        self.assertEqual(tm.get_active_tracks()[0].landing_counter, 0)
        self.assertEqual(tm.get_active_tracks()[0].lifecycle_state, LifecycleState.FALLING)

    # 13. Duplicate incident prevention
    def test_13_duplicate_incident_prevention(self) -> None:
        corridor = ConfigManager.build_amr_corridor(self.default_config)
        ev_logger = EvidenceLogger(
            self.default_config.storage, corridor, self.default_config.amr
        )
        evaluator = RiskEvaluator(self.default_config.risk, corridor)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        poly = box(300.0, 400.0, 340.0, 440.0)
        risk = evaluator.evaluate(poly)

        from track_manager import TrackedObject

        track = TrackedObject(
            track_id=42,
            external_tracker_id=1,
            class_id=0,
            class_name="box",
            confidence=0.95,
            bbox=(300.0, 400.0, 340.0, 440.0),
            centroid=(320.0, 420.0),
            footprint=poly,
            has_segmentation_mask=False,
            mask_contour=None,
        )

        id1 = ev_logger.start_incident(
            track, risk, frame_id=100, current_frame=dummy_frame, pre_event_frames=[dummy_frame], post_event_frames_override=0
        )
        self.assertIsNotNone(id1)

        # Second call for the same track_id must return None
        id2 = ev_logger.start_incident(
            track, risk, frame_id=101, current_frame=dummy_frame, pre_event_frames=[dummy_frame], post_event_frames_override=0
        )
        self.assertIsNone(id2)
        ev_logger.flush_and_shutdown()

    # 14. Segmentation footprint extraction & 15. Bounding-box fallback
    def test_14_15_segmentation_footprint_and_bbox_fallback(self) -> None:
        # Valid segmentation mask contour
        valid_mask = np.array(
            [[300.0, 400.0], [340.0, 400.0], [340.0, 450.0], [300.0, 450.0]],
            dtype=np.float64,
        )
        bbox = (295.0, 395.0, 345.0, 455.0)
        poly, used_mask, contour = ObjectDetector.extract_footprint_from_mask_or_bbox(
            valid_mask, bbox
        )
        self.assertTrue(used_mask)
        self.assertIsNotNone(poly)
        self.assertIsNotNone(contour)
        assert poly is not None
        self.assertAlmostEqual(poly.area, 2000.0, places=2)

        # Missing or degenerate segmentation mask (only 2 points) -> falls back to bbox
        degenerate_mask = np.array([[300.0, 400.0], [340.0, 400.0]], dtype=np.float64)
        poly_fb, used_mask_fb, _ = ObjectDetector.extract_footprint_from_mask_or_bbox(
            degenerate_mask, bbox
        )
        self.assertFalse(used_mask_fb)
        self.assertIsNotNone(poly_fb)
        assert poly_fb is not None
        self.assertAlmostEqual(poly_fb.area, 50.0 * 60.0, places=2)

    # 16. HIGH classification, 17. MEDIUM classification, 18. LOW classification,
    # 19. Boundary contact, 20. UNCERTAIN geometry
    def test_16_17_18_19_20_risk_classifications(self) -> None:
        config = ConfigManager.load(
            self.config_path,
            overrides={
                "corridor.yellow_line_p1": "320,100",
                "corridor.yellow_line_p2": "320,900",
                "amr.base_half_width": "40",
                "amr.safety_margin": "20",
            },
        )
        # Corridor spans x in [260, 380], y in [100, 900]
        corridor = ConfigManager.build_amr_corridor(config)
        evaluator = RiskEvaluator(config.risk, corridor)

        # 16. HIGH risk: Box completely inside corridor (100% overlap >= 10%)
        high_poly = box(300.0, 400.0, 340.0, 440.0)
        res_high = evaluator.evaluate(high_poly)
        self.assertEqual(res_high.risk_level, RiskLevel.HIGH)
        self.assertAlmostEqual(res_high.overlap_percentage, 100.0, places=2)
        self.assertTrue(res_high.geometry_valid)

        # 17. MEDIUM risk: Box of width 100 from x=165 to x=265 -> overlaps x in [260, 265] = 5% (< 10%)
        med_poly = box(165.0, 400.0, 265.0, 440.0)
        res_med = evaluator.evaluate(med_poly)
        self.assertEqual(res_med.risk_level, RiskLevel.MEDIUM)
        self.assertAlmostEqual(res_med.overlap_percentage, 5.0, places=2)

        # Exactly 10% is HIGH under the configured >= threshold.
        ten_percent_poly = box(170.0, 400.0, 270.0, 440.0)
        res_ten_percent = evaluator.evaluate(ten_percent_poly)
        self.assertEqual(res_ten_percent.risk_level, RiskLevel.HIGH)
        self.assertAlmostEqual(res_ten_percent.overlap_percentage, 10.0, places=2)

        # 18. LOW risk: Box completely outside corridor (x in [50, 100])
        low_poly = box(50.0, 400.0, 100.0, 440.0)
        res_low = evaluator.evaluate(low_poly)
        self.assertEqual(res_low.risk_level, RiskLevel.LOW)
        self.assertAlmostEqual(res_low.overlap_percentage, 0.0, places=2)
        self.assertAlmostEqual(res_low.external_distance, 160.0, places=2)

        res_pass_through = evaluator.evaluate(low_poly, corridor_contact_history=True)
        self.assertEqual(res_pass_through.risk_level, RiskLevel.LOW)
        self.assertGreater(res_pass_through.external_distance, 0.0)

        # 19. Boundary contact: Box from x=200 to x=260 touches corridor left edge x=260
        touch_poly = box(200.0, 400.0, 260.0, 440.0)
        res_touch = evaluator.evaluate(touch_poly)
        self.assertEqual(res_touch.risk_level, RiskLevel.MEDIUM)
        self.assertTrue(res_touch.touches_boundary)
        self.assertAlmostEqual(res_touch.overlap_percentage, 0.0, places=2)

        # 20. UNCERTAIN geometry: None, empty polygon, self-intersecting bowtie polygon
        res_none = evaluator.evaluate(None)
        self.assertEqual(res_none.risk_level, RiskLevel.UNCERTAIN)
        self.assertFalse(res_none.geometry_valid)

        bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
        res_invalid = evaluator.evaluate(bowtie)
        self.assertEqual(res_invalid.risk_level, RiskLevel.UNCERTAIN)
        self.assertNotEqual(res_invalid.risk_level, RiskLevel.LOW)

    # 21. Circular frame buffer, 22. Pre-event recording, 23. Post-event recording,
    # 24. Snapshot creation, 25. MP4 creation, 26. JSON validation
    def test_21_to_26_buffer_and_evidence_artifacts(self) -> None:
        corridor = ConfigManager.build_amr_corridor(self.default_config)
        ev_logger = EvidenceLogger(
            self.default_config.storage, corridor, self.default_config.amr
        )
        evaluator = RiskEvaluator(self.default_config.risk, corridor)

        from track_manager import TrackedObject

        poly = box(300.0, 400.0, 340.0, 440.0)
        risk = evaluator.evaluate(poly)
        track = TrackedObject(
            track_id=7,
            external_tracker_id=7,
            class_id=0,
            class_name="box",
            confidence=0.94,
            bbox=(300.0, 400.0, 340.0, 440.0),
            centroid=(320.0, 420.0),
            footprint=poly,
            has_segmentation_mask=True,
            mask_contour=[(300.0, 400.0), (340.0, 400.0), (340.0, 440.0), (300.0, 440.0)],
        )

        # Create 10 pre-event frames and 5 post-event frames
        pre_frames = [
            np.full((480, 640, 3), fill_value=i * 10, dtype=np.uint8) for i in range(10)
        ]
        landing_frame = np.full((480, 640, 3), fill_value=128, dtype=np.uint8)

        event_id = ev_logger.start_incident(
            track=track,
            risk_result=risk,
            frame_id=250,
            current_frame=landing_frame,
            pre_event_frames=pre_frames,
            is_preexisting_debris=False,
            post_event_frames_override=5,
        )
        self.assertIsNotNone(event_id)

        for p in range(5):
            post_frm = np.full((480, 640, 3), fill_value=200 + p, dtype=np.uint8)
            ev_logger.ingest_post_event_frame(post_frm)

        completed = ev_logger.flush_and_shutdown()
        self.assertEqual(len(completed), 1)
        rec = completed[0]
        self.assertTrue(rec.success)
        self.assertTrue(rec.snapshot_path.exists())
        self.assertTrue(rec.video_path.exists())
        self.assertTrue(rec.metadata_path.exists())

        # Verify snapshot.jpg is readable
        snap_img = cv2.imread(str(rec.snapshot_path))
        self.assertIsNotNone(snap_img)
        self.assertEqual(snap_img.shape, (480, 640, 3))

        # Verify incident_clip.mp4 has 10 pre + 5 post = 15 frames
        cap = cv2.VideoCapture(str(rec.video_path))
        self.assertTrue(cap.isOpened())
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.assertEqual(frame_cnt, 15)

        # Verify metadata.json schema and values
        meta = json.loads(rec.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["event_id"], event_id)
        self.assertEqual(meta["frame_id"], 250)
        self.assertEqual(meta["detected_class"], "box")
        self.assertEqual(meta["track_id"], 7)
        self.assertAlmostEqual(meta["confidence"], 0.94, places=2)
        self.assertFalse(meta["flags"]["is_preexisting_debris"])
        self.assertTrue(meta["flags"]["fallen_on_to_ground"])
        self.assertEqual(meta["risk_assessment"]["risk_level"], "HIGH")
        self.assertAlmostEqual(meta["spatial_analysis"]["overlap_percentage"], 100.0, places=2)
        self.assertEqual(meta["evidence_files"]["snapshot"], "snapshot.jpg")
        self.assertEqual(meta["evidence_files"]["video_clip"], "incident_clip.mp4")

    # 27. Synthetic-video end-to-end integration
    def test_27_synthetic_video_end_to_end_integration(self) -> None:
        video_path = self.temp_dir / "synthetic_warehouse.mp4"
        evidence_dir = self.temp_dir / "e2e_evidence"

        # Generate a 45-frame synthetic warehouse video (640x960 @ 30 FPS)
        width, height, fps, total_frames = 640, 960, 30, 45
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (width, height))
        self.assertTrue(writer.isOpened())

        frame_plan: Dict[int, FakeResult] = {}

        # Stationary debris at (80, 500, 130, 550) present in every frame
        debris_box = (80.0, 500.0, 130.0, 550.0)
        debris_mask = np.array(
            [[80.0, 500.0], [130.0, 500.0], [130.0, 550.0], [80.0, 550.0]],
            dtype=np.float64,
        )

        # Falling box appears at frame 12 (after 0.3s baseline observation window)
        # Falls from y=150 down to y=450 inside the AMR corridor (x in [300, 340]),
        # then rests from frame 22 to frame 45 (24 resting frames >= 15 rest_frames)
        fall_y = 150.0
        for f_idx in range(1, total_frames + 1):
            img = np.full((height, width, 3), 40, dtype=np.uint8)
            # Draw stationary debris
            cv2.rectangle(img, (80, 500), (130, 550), (120, 120, 120), -1)

            boxes_xyxy = [debris_box]
            confs = [0.91]
            clss = [0]
            masks_xy = [debris_mask]

            if f_idx >= 12:
                if f_idx < 22:
                    fall_y += 30.0  # 30 px/frame @ 30 FPS = 900 px/s downward
                f_box = (300.0, fall_y, 340.0, fall_y + 40.0)
                f_mask = np.array(
                    [
                        [300.0, fall_y],
                        [340.0, fall_y],
                        [340.0, fall_y + 40.0],
                        [300.0, fall_y + 40.0],
                    ],
                    dtype=np.float64,
                )
                cv2.rectangle(
                    img,
                    (300, int(fall_y)),
                    (340, int(fall_y + 40)),
                    (0, 140, 255),
                    -1,
                )
                boxes_xyxy.append(f_box)
                confs.append(0.96)
                clss.append(0)
                masks_xy.append(f_mask)

            writer.write(img)
            frame_plan[f_idx] = FakeResult(
                boxes=FakeBoxes(boxes_xyxy, confs, clss),
                masks=FakeMasks(masks_xy),
            )

        writer.release()

        overrides = {
            "input.source": "video",
            "input.video_path": str(video_path),
            "corridor.yellow_line_p1": "320,100",
            "corridor.yellow_line_p2": "320,900",
            "baseline.enabled": "true",
            "baseline.observation_duration_sec": "0.30",
            "baseline.minimum_observations": "5",
            "storage.evidence_base_dir": str(evidence_dir),
            "storage.buffer_duration_sec": "1.0",
            "storage.post_event_duration_sec": "0.2",
            "display.enabled": "false",
        }
        cfg = ConfigManager.load(self.config_path, overrides=overrides)
        mock_detector = ObjectDetector(
            config=cfg.model,
            geometry_epsilon=cfg.risk.geometry_epsilon,
            model_instance=ScriptedYoloModel(frame_plan, task="segment"),
        )

        app = DownfallHazardApplication(
            config_path=self.config_path,
            cli_overrides=overrides,
            injected_detector=mock_detector,
        )
        exit_code = app.run()
        self.assertEqual(exit_code, 0)

        # Verify that stationary debris was registered and did NOT trigger an incident,
        # while the falling box triggered exactly 1 HIGH risk incident with all 3 files
        self.assertEqual(len(app.incident_registry), 1)
        self.assertEqual(app.telemetry.processed_frames, total_frames)

        run_folders = [p for p in evidence_dir.iterdir() if p.is_dir()]
        self.assertEqual(len(run_folders), 1)
        run_dir = run_folders[0]
        annotated_video = run_dir / "annotated_video.mp4"
        self.assertTrue(annotated_video.exists())
        cap = cv2.VideoCapture(str(annotated_video))
        self.assertTrue(cap.isOpened())
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), total_frames)
        cap.release()

        run_metadata = json.loads(
            (run_dir / "run_metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(run_metadata["processed_frames"], total_frames)
        self.assertEqual(run_metadata["annotated_video_frames"], total_frames)

        incident_folders = [p for p in (run_dir / "incidents").iterdir() if p.is_dir()]
        self.assertEqual(len(incident_folders), 1)
        inc_dir = incident_folders[0]
        self.assertTrue((inc_dir / "snapshot.jpg").exists())
        self.assertTrue((inc_dir / "incident_clip.mp4").exists())
        self.assertTrue((inc_dir / "metadata.json").exists())

        meta = json.loads((inc_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["risk_assessment"]["risk_level"], "HIGH")
        self.assertFalse(meta["flags"]["is_preexisting_debris"])
        self.assertTrue(meta["flags"]["fallen_on_to_ground"])

    def test_28_isaac_sim_ros2_unavailable_handling(self) -> None:
        """Verify IsaacSimRos2FrameSource fails cleanly when ROS 2 is unavailable."""
        source = IsaacSimRos2FrameSource(
            topic="/camera/rgb/image_raw",
            expected_encoding="bgr8",
            reconnect_enabled=False,
            reconnect_attempts=1,
        )
        with self.assertRaises(FrameSourceError):
            source.open()


if __name__ == "__main__":
    unittest.main(verbosity=2)
