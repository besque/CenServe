# shared_types.py
from dataclasses import dataclass, field
from typing import Tuple, Literal

# Output delay (seconds). Used by VideoLoop/VirtualVideoLoop only.
# Web server: set to 0 for real-time (no buffer). Text PII blur may appear 1–2 s late.
AV_DELAY_SECONDS = 0.0

# How often each detector receives a new frame (every N frames).
# Lower = more responsive but heavier on CPU.
DETECTION_CADENCE = {
    'face':             3,
    'objects_camera':  8,   # Increased from 10 for better responsiveness
    'objects_screen':  10,  # Increased from 15 for better streaming
    'nsfw':             8,  # Increased from 15 for better responsiveness
    'text_pii_camera': 20,
    'text_pii_screen':  5,   # 5 = request new OCR more often (blur updates a bit snappier)
}

# Cached detection results older than this many frames are discarded.
# At 30 fps, 30 frames ≈ 1 second — short enough that blur doesn't
# stick to a region after content moves, but long enough for smooth
# coverage between detector runs.
CACHE_TTL_FRAMES = 30

@dataclass
class DetectionEvent:
    type: Literal["face", "plate", "card", "nsfw", "text_pii"]
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in ORIGINAL frame pixels
    confidence: float                # 0.0 to 1.0
    frame_id: int
    blur: bool = True               # False = whitelisted (don't blur)

@dataclass
class PipelineConfig:
    blur_faces: bool = True
    blur_plates: bool = True
    blur_cards: bool = True
    blur_nsfw: bool = True
    blur_text_pii: bool = True
    detection_cadence: int = 15    
    blur_strength: int = 51         # gaussian kernel size (must be odd number)
    face_similarity_threshold: float = 0.4