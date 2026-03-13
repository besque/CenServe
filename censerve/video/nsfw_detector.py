import numpy as np
from nudenet import NudeDetector
from typing import List
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared_types import DetectionEvent

# NudeNet labels to blur — comprehensive list for debugging
LABELS_TO_BLUR = [
    # Primary explicit content
    "EXPOSED_BREAST_F",
    "EXPOSED_GENITALIA_F", 
    "EXPOSED_GENITALIA_M",
    "EXPOSED_ANUS_F",
    "EXPOSED_ANUS_M",
    "EXPOSED_BUTTOCKS",
    
    # Common NudeNet labels (enable all for debugging)
    "BELLY",
    "FEET_F",
    "FEET_M", 
    "ARMPITS_F",
    "ARMPITS_M",
    "FACE_F",
    "FACE_M",
    "COVERED_GENITALIA_F",
    "COVERED_GENITALIA_M",
    "COVERED_BREAST_F",
    "COVERED_BUTTOCKS",
    
    # Other possible labels
    "FEMALE_BREAST_EXPOSED",
    "MALE_GENITALIA_EXPOSED", 
    "FEMALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
]

class NSFWDetector:
    def __init__(self, confidence_threshold: float = 0.1):  # Very low for debugging
        self.conf = confidence_threshold
        # NudeNet downloads model automatically on first use (~15MB)
        self.detector = NudeDetector()
        print("[NSFWDetector] NudeNet loaded")

    def detect(self, frame: np.ndarray, frame_id: int) -> List[DetectionEvent]:
        """Returns DetectionEvents for NSFW regions."""
        try:
            # NudeNet expects a file path or numpy array
            detections = self.detector.detect(frame)
            
        except Exception as e:
            print(f"[NSFW] ERROR during detection: {e}")
            return []
        
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