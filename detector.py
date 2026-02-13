from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO  # type: ignore

    _YOLO_AVAILABLE = True
except Exception:
    YOLO = None  # type: ignore
    _YOLO_AVAILABLE = False


@dataclass
class Detection:
    """Represents a single face/head detection."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    label: str = "face"


class FaceDetector:
    """
    Enhanced detector for faces and heads from various angles.
    Optimized for exam scenarios with bowed faces and back views.

    - model_type="yolo": use YOLOv8 via `ultralytics` (requires Torch).
      Detects persons and extracts head regions, works from all angles.
    - model_type="haar": use OpenCV Haar Cascade with multiple profiles.
      Detects frontal and profile faces.
    """

    def __init__(
        self,
        model_type: str = "yolo",
        yolo_model_path: str | None = "yolov8n.pt",
        confidence: float = 0.3,  # Lower default for better recall
        head_detection_mode: bool = True,  # Focus on head regions
    ) -> None:
        self.model_type = model_type if model_type in {"yolo", "haar"} else "haar"
        self.confidence = float(confidence)
        self.head_detection_mode = head_detection_mode

        self._yolo_model = None
        self._haar_cascade_frontal = None
        self._haar_cascade_profile = None

        if self.model_type == "yolo" and _YOLO_AVAILABLE:
            try:
                # YOLOv8 detects persons (class 0), we'll extract head regions
                self._yolo_model = YOLO(yolo_model_path)  # type: ignore[arg-type]
            except Exception:
                # Gracefully fall back to Haar cascade if YOLO fails to load.
                self.model_type = "haar"

        if self.model_type == "haar":
            # Load multiple cascade profiles for different angles
            frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
            self._haar_cascade_frontal = cv2.CascadeClassifier(frontal_path)
            self._haar_cascade_profile = cv2.CascadeClassifier(profile_path)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Run head/face detection on a single BGR frame.
        Optimized for detecting heads from various angles including bowed faces and back views.

        Returns a list of Detection objects.
        """
        if self.model_type == "yolo" and self._yolo_model is not None:
            return self._detect_yolo(frame_bgr)
        return self._detect_haar(frame_bgr)

    def _detect_yolo(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Detect persons using YOLO and extract head regions.
        Works from all angles including back views and bowed positions.
        """
        h, w = frame_bgr.shape[:2]
        # Use lower confidence for better recall in exam scenarios
        results = self._yolo_model(  # type: ignore[operator]
            frame_bgr, conf=max(0.1, self.confidence * 0.7), verbose=False
        )
        detections: List[Detection] = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                # In the default COCO model, class 0 is "person"
                if cls_id != 0:
                    continue
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                if conf < self.confidence * 0.7:  # Lower threshold for recall
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Clip to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                
                if self.head_detection_mode:
                    # Extract head region from person detection
                    # Head is typically in the upper 1/3 to 1/2 of person bounding box
                    person_height = y2 - y1
                    head_height = int(person_height * 0.35)  # Upper 35% for head
                    head_y1 = y1
                    head_y2 = y1 + head_height
                    
                    # Widen head box slightly to account for side views
                    person_width = x2 - x1
                    head_width = int(person_width * 1.1)  # 10% wider
                    head_x_center = (x1 + x2) // 2
                    head_x1 = max(0, head_x_center - head_width // 2)
                    head_x2 = min(w - 1, head_x_center + head_width // 2)
                    
                    detections.append(
                        Detection(
                            x1=head_x1,
                            y1=head_y1,
                            x2=head_x2,
                            y2=head_y2,
                            score=conf,
                            label="head",
                        )
                    )
                else:
                    # Full person detection
                    detections.append(
                        Detection(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            score=conf,
                            label="person",
                        )
                    )
        
        # Remove overlapping detections (NMS-like filtering)
        return self._filter_overlaps(detections)

    def _detect_haar(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Detect faces using multiple Haar cascade profiles.
        Tries frontal and profile views for better coverage.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        detections: List[Detection] = []
        
        # Detect frontal faces with relaxed parameters for bowed faces
        if self._haar_cascade_frontal is not None:
            frontal_faces = self._haar_cascade_frontal.detectMultiScale(
                gray,
                scaleFactor=1.15,  # Slightly more aggressive scaling
                minNeighbors=4,   # Lower for better recall
                minSize=(30, 30),  # Smaller minimum size for bowed faces
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            for (x, y, w, h) in frontal_faces:
                detections.append(
                    Detection(
                        x1=int(x),
                        y1=int(y),
                        x2=int(x + w),
                        y2=int(y + h),
                        score=0.8,
                        label="face_frontal",
                    )
                )
        
        # Detect profile faces (side views)
        if self._haar_cascade_profile is not None:
            profile_faces = self._haar_cascade_profile.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=4,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            for (x, y, w, h) in profile_faces:
                detections.append(
                    Detection(
                        x1=int(x),
                        y1=int(y),
                        x2=int(x + w),
                        y2=int(y + h),
                        score=0.7,
                        label="face_profile",
                    )
                )
        
        # Remove overlapping detections
        return self._filter_overlaps(detections)
    
    def _filter_overlaps(self, detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
        """
        Remove overlapping detections using IoU (Intersection over Union).
        Keeps the detection with higher confidence.
        """
        if len(detections) <= 1:
            return detections
        
        def calculate_iou(box1: Detection, box2: Detection) -> float:
            """Calculate Intersection over Union between two boxes."""
            x1_inter = max(box1.x1, box2.x1)
            y1_inter = max(box1.y1, box2.y1)
            x2_inter = min(box1.x2, box2.x2)
            y2_inter = min(box1.y2, box2.y2)
            
            if x2_inter <= x1_inter or y2_inter <= y1_inter:
                return 0.0
            
            inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
            box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
            union_area = box1_area + box2_area - inter_area
            
            if union_area == 0:
                return 0.0
            return inter_area / union_area
        
        # Sort by confidence (descending)
        sorted_detections = sorted(detections, key=lambda d: d.score, reverse=True)
        filtered: List[Detection] = []
        
        for det in sorted_detections:
            is_duplicate = False
            for kept in filtered:
                if calculate_iou(det, kept) > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(det)
        
        return filtered


def draw_detections(
    frame_bgr: np.ndarray,
    detections: List[Detection],
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw bounding boxes and labels on a BGR frame and return a copy."""
    output = frame_bgr.copy()
    for i, det in enumerate(detections, start=1):
        cv2.rectangle(output, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        label = f"{det.label} #{i} ({det.score:.2f})"
        cv2.putText(
            output,
            label,
            (det.x1, max(det.y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return output

