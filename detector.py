"""YOLO26 object detection and segmentation footprint extraction module."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon, box

from config_manager import YoloConfig

logger = logging.getLogger(__name__)


class DetectorError(RuntimeError):
    """Raised when the YOLO detector cannot be initialized or fails fatally."""


@dataclass(frozen=True)
class Detection:
    """Validated single object detection with bounding box and footprint polygon."""

    detection_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    footprint: Polygon
    has_segmentation_mask: bool
    mask_contour: Optional[List[Tuple[float, float]]] = None


@dataclass(frozen=True)
class DetectionFrameOutput:
    """Structured output returned by ObjectDetector to main.py for a single frame."""

    frame_id: int
    timestamp: float
    detections: List[Detection]
    inference_latency_ms: float
    used_segmentation: bool


class ObjectDetector:
    """
    Wraps YOLO26 detection and segmentation inference.

    Extracts class ID, class name, confidence, bounding box, centroid, and
    segmentation mask polygon (with automatic fallback to bounding-box polygon
    when masks are unavailable or invalid).
    """

    def __init__(
        self,
        config: YoloConfig,
        geometry_epsilon: float = 1e-6,
        model_instance: Optional[Any] = None,
    ) -> None:
        self._config = config
        self._epsilon = geometry_epsilon
        self._model: Any = model_instance
        self._device: str = str(config.device)
        self._supports_segmentation: bool = False
        self._class_names: Dict[int, str] = {}

        if self._model is None:
            self._load_model()
        else:
            self._inspect_injected_model()

    def _load_model(self) -> None:
        """Load the YOLO26 checkpoint via Ultralytics with hardware fallback handling."""
        checkpoint_path = Path(self._config.path)

        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError(
                f"Ultralytics / PyTorch dependency is unavailable: {exc}"
            ) from exc

        # Resolve device availability gracefully
        requested_device = str(self._config.device).strip()
        if requested_device.lower() != "cpu":
            if not torch.cuda.is_available():
                logger.warning(
                    "Configured GPU device '%s' is unavailable; falling back to CPU.",
                    requested_device,
                )
                self._device = "cpu"
            else:
                self._device = requested_device
        else:
            self._device = "cpu"

        try:
            self._model = YOLO(str(checkpoint_path))
            task = getattr(self._model, "task", "detect")
            if task not in ("detect", "segment"):
                raise DetectorError(
                    f"Unsupported YOLO model task '{task}'. Only 'detect' and 'segment' are supported."
                )
            self._supports_segmentation = task == "segment"
            names = getattr(self._model, "names", {})
            if isinstance(names, dict):
                self._class_names = {int(k): str(v) for k, v in names.items()}
            elif isinstance(names, (list, tuple)):
                self._class_names = {idx: str(v) for idx, v in enumerate(names)}
        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(f"Failed to load YOLO model '{checkpoint_path}': {exc}") from exc

    def _inspect_injected_model(self) -> None:
        """Inspect a pre-loaded or mock model instance."""
        task = getattr(self._model, "task", "detect")
        if task not in ("detect", "segment"):
            raise DetectorError(
                f"Unsupported YOLO model task '{task}'. Only 'detect' and 'segment' are supported."
            )
        self._supports_segmentation = task == "segment"
        names = getattr(self._model, "names", {})
        if isinstance(names, dict):
            self._class_names = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, (list, tuple)):
            self._class_names = {idx: str(v) for idx, v in enumerate(names)}

    @staticmethod
    def build_bbox_polygon(
        bbox: Tuple[float, float, float, float], epsilon: float = 1e-6
    ) -> Optional[Polygon]:
        """Build and validate a Shapely Polygon from a bounding box (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = bbox
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            return None
        if (x2 - x1) <= epsilon or (y2 - y1) <= epsilon:
            return None
        poly = box(x1, y1, x2, y2)
        if not poly.is_valid or poly.is_empty or poly.area <= epsilon:
            return None
        return poly

    @classmethod
    def extract_footprint_from_mask_or_bbox(
        cls,
        mask_xy: Optional[ np.ndarray | List[Tuple[float, float]] ],
        bbox: Tuple[float, float, float, float],
        epsilon: float = 1e-6,
    ) -> Tuple[Optional[Polygon], bool, Optional[List[Tuple[float, float]]]]:
        """
        Attempt to build a valid Shapely Polygon from segmentation contour coordinates.
        If the mask is absent, degenerate, self-intersecting beyond repair, or zero-area,
        fall back to the bounding-box polygon.
        """
        if mask_xy is not None:
            try:
                pts_array = np.asarray(mask_xy, dtype=np.float64)
                if pts_array.ndim == 2 and pts_array.shape[0] >= 3 and pts_array.shape[1] == 2:
                    if np.all(np.isfinite(pts_array)):
                        contour = [(float(pt[0]), float(pt[1])) for pt in pts_array]
                        poly = Polygon(contour)
                        if not poly.is_valid:
                            repaired = poly.buffer(0)
                            if isinstance(repaired, Polygon):
                                poly = repaired
                            elif hasattr(repaired, "geoms"):
                                polys = [
                                    g for g in repaired.geoms
                                    if isinstance(g, Polygon) and g.area > epsilon
                                ]
                                if polys:
                                    poly = max(polys, key=lambda p: p.area)
                        if isinstance(poly, Polygon) and poly.is_valid and not poly.is_empty and poly.area > epsilon:
                            ext_coords = [(float(c[0]), float(c[1])) for c in poly.exterior.coords[:-1]]
                            return poly, True, ext_coords
            except Exception as exc:
                logger.debug("Segmentation polygon extraction failed, falling back to bbox: %s", exc)

        fallback_poly = cls.build_bbox_polygon(bbox, epsilon=epsilon)
        return fallback_poly, False, None

    def detect(self, frame: np.ndarray, frame_id: int, timestamp: float) -> DetectionFrameOutput:
        """
        Execute YOLO26 inference on a BGR frame and return validated Detection objects.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid or empty frame provided to ObjectDetector.detect().")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected 3-channel BGR image, got shape {frame.shape}.")

        start_time = time.perf_counter()
        try:
            results = self._model.predict(
                source=frame,
                conf=self._config.conf_threshold,
                iou=self._config.iou_threshold,
                imgsz=self._config.image_size,
                device=self._device,
                verbose=False,
            )
        except RuntimeError as exc:
            err_msg = str(exc).lower()
            if "out of memory" in err_msg or "cuda" in err_msg:
                logger.error("CUDA error during inference (%s); retrying on CPU.", exc)
                self._device = "cpu"
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                results = self._model.predict(
                    source=frame,
                    conf=self._config.conf_threshold,
                    iou=self._config.iou_threshold,
                    imgsz=self._config.image_size,
                    device="cpu",
                    verbose=False,
                )
            else:
                raise DetectorError(f"YOLO inference runtime failure: {exc}") from exc
        except Exception as exc:
            raise DetectorError(f"YOLO inference failure: {exc}") from exc

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        validated_detections: List[Detection] = []
        any_segmentation_used = False

        if not results:
            return DetectionFrameOutput(
                frame_id=frame_id,
                timestamp=timestamp,
                detections=validated_detections,
                inference_latency_ms=latency_ms,
                used_segmentation=False,
            )

        result = results[0]
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)

        if boxes is None or len(boxes) == 0:
            return DetectionFrameOutput(
                frame_id=frame_id,
                timestamp=timestamp,
                detections=validated_detections,
                inference_latency_ms=latency_ms,
                used_segmentation=False,
            )

        xyxy_arr = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        conf_arr = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        cls_arr = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)

        mask_polygons: Optional[List[Any]] = None
        if masks is not None and hasattr(masks, "xy") and masks.xy is not None:
            mask_polygons = list(masks.xy)

        result_names = getattr(result, "names", None) or self._class_names

        det_counter = 0
        for idx in range(len(xyxy_arr)):
            try:
                x1, y1, x2, y2 = [float(v) for v in xyxy_arr[idx]]
                conf = float(conf_arr[idx])
                cls_id = int(cls_arr[idx])
            except (ValueError, TypeError, IndexError):
                logger.warning("Skipping malformed detection at index %d in frame %d", idx, frame_id)
                continue

            if not all(math.isfinite(v) for v in (x1, y1, x2, y2, conf)):
                logger.warning("Skipping non-finite detection at index %d in frame %d", idx, frame_id)
                continue

            if conf < self._config.conf_threshold or conf > 1.0:
                continue

            if (x2 - x1) <= self._epsilon or (y2 - y1) <= self._epsilon:
                logger.warning("Skipping zero/negative-area bbox at index %d in frame %d", idx, frame_id)
                continue

            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0
            bbox_tuple = (x1, y1, x2, y2)

            raw_mask = None
            if mask_polygons is not None and idx < len(mask_polygons):
                raw_mask = mask_polygons[idx]

            footprint, used_mask, contour = self.extract_footprint_from_mask_or_bbox(
                raw_mask, bbox_tuple, epsilon=self._epsilon
            )
            if footprint is None:
                logger.warning("Skipping detection with invalid footprint at index %d in frame %d", idx, frame_id)
                continue

            if used_mask:
                any_segmentation_used = True

            if isinstance(result_names, dict):
                cls_name = str(result_names.get(cls_id, f"class_{cls_id}"))
            elif isinstance(result_names, (list, tuple)) and 0 <= cls_id < len(result_names):
                cls_name = str(result_names[cls_id])
            else:
                cls_name = f"class_{cls_id}"

            det_counter += 1
            validated_detections.append(
                Detection(
                    detection_id=det_counter,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=bbox_tuple,
                    centroid=(xc, yc),
                    footprint=footprint,
                    has_segmentation_mask=used_mask,
                    mask_contour=contour,
                )
            )

        return DetectionFrameOutput(
            frame_id=frame_id,
            timestamp=timestamp,
            detections=validated_detections,
            inference_latency_ms=latency_ms,
            used_segmentation=any_segmentation_used,
        )
