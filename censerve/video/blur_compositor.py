import cv2
import numpy as np
from typing import List
from shared_types import DetectionEvent, PipelineConfig

def apply_blurs(frame: np.ndarray, events: List[DetectionEvent], config: PipelineConfig) -> np.ndarray:
    """Apply gaussian blur to all detection bounding boxes on the frame."""
    output = frame.copy()
    
    for event in events:
        if not event.blur:
            continue  # whitelisted face — skip
            
        # Check if this event type is enabled in config
        type_enabled = {
            "face": config.blur_faces,
            "plate": config.blur_plates,
            "card": config.blur_cards,
            "nsfw": config.blur_nsfw,
            "text_pii": config.blur_text_pii,
        }
        if not type_enabled.get(event.type, True):
            continue
        
        x1, y1, x2, y2 = event.bbox
        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        roi = output[y1:y2, x1:x2]
        k = config.blur_strength
        # kernel must be odd
        k = k if k % 2 == 1 else k + 1
        blurred_roi = cv2.GaussianBlur(roi, (k, k), 0)
        output[y1:y2, x1:x2] = blurred_roi
    
    return output


def draw_debug_overlay(frame: np.ndarray, events: List[DetectionEvent]) -> np.ndarray:
    """Draw bounding boxes and labels for debugging. (disable in final demo.)"""
    colors = {
        "face": (0, 255, 0),
        "plate": (255, 165, 0),
        "card": (255, 0, 0),
        "nsfw": (0, 0, 255),
        "text_pii": (255, 255, 0),
    }
    output = frame.copy()
    for event in events:
        x1, y1, x2, y2 = event.bbox
        color = colors.get(event.type, (255, 255, 255))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{event.type} {event.confidence:.2f}"
        cv2.putText(output, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return output