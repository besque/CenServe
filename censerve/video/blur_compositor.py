import cv2
import numpy as np
from typing import List
from shared_types import DetectionEvent, PipelineConfig

def _blur_region(output: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                 k: int, oval: bool = False) -> None:
    """Blur a region. If oval=True, use elliptical mask (better for faces)."""
    h, w = output.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = output[y1:y2, x1:x2].copy()
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    if oval:
        kh, kw = roi.shape[:2]
        mask = np.zeros((kh, kw), dtype=np.uint8)
        center = (kw // 2, kh // 2)
        axes = (kw // 2, kh // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask_3ch = np.expand_dims(mask, axis=2)
        roi = np.where(mask_3ch > 0, blurred, roi)
    else:
        roi = blurred
    output[y1:y2, x1:x2] = roi


def apply_blurs(frame: np.ndarray, events: List[DetectionEvent], config: PipelineConfig) -> np.ndarray:
    """Apply gaussian blur to all detection bounding boxes on the frame."""
    output = frame.copy()
    k = config.blur_strength
    k = k if k % 2 == 1 else k + 1

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
        oval = event.type == "face"
        _blur_region(output, x1, y1, x2, y2, k, oval=oval)

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