import numpy as np
from nudenet import NudeDetector
from typing import List
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared_types import DetectionEvent

# NudeNet labels to blur — adjust based on your use case
LABELS_TO_BLUR = [
    "EXPOSED_BREAST_F",
    "EXPOSED_GENITALIA_F",
    "EXPOSED_GENITALIA_M",
    "EXPOSED_ANUS_F",
    "EXPOSED_ANUS_M",
    "EXPOSED_BUTTOCKS",
    # Add "FACE_F", "FACE_M" if you want faces blurred via NudeNet too (skip, MediaPipe handles faces)
]

class NSFWDetector:
    def __init__(self, confidence_threshold: float = 0.4):
        self.conf = confidence_threshold
        # NudeNet downloads model automatically on first use (~15MB)
        self.detector = NudeDetector()
        print("[NSFWDetector] NudeNet loaded")

    def detect(self, frame: np.ndarray, frame_id: int) -> List[DetectionEvent]:
        """Returns DetectionEvents for NSFW regions."""
        # NudeNet expects a file path or numpy array
        detections = self.detector.detect(frame)
        
        events = []
        for det in detections:
            if det["score"] < self.conf:
                continue
            if det["class"] not in LABELS_TO_BLUR:
                continue
            
            box = det["box"]  # [x, y, w, h]
            x1, y1, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x2, y2 = x1 + w, y1 + h
            
            events.append(DetectionEvent(
                type="nsfw",
                bbox=(x1, y1, x2, y2),
                confidence=float(det["score"]),
                frame_id=frame_id,
                blur=True
            ))
        
        return events


def make_nsfw_detector(**kwargs):
    detector = NSFWDetector(**kwargs)
    return detector.detect