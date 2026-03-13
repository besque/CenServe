# shared_types.py — Person 1 creates this file
from dataclasses import dataclass, field
from typing import Tuple, Literal

# Set to 0 for real-time testing (no intentional AV delay)
AV_DELAY_SECONDS = 0.0

# How often each detector receives a new frame (every N frames).
# Lower = more responsive but heavier on CPU.
DETECTION_CADENCE = {
    'face':             3,
    'objects_camera':  10,
    'objects_screen':  15,
    'nsfw':            15,
    'text_pii_camera': 20,
    'text_pii_screen':  8,
}

# Cached detection results older than this many frames are discarded.
# At 30 fps, 45 frames ≈ 1.5 seconds — enough to avoid stale blur
# sticking to a region after content moves.
CACHE_TTL_FRAMES = 45

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
    detection_cadence: int = 15     # run detection every N frames
    blur_strength: int = 51         # gaussian kernel size (must be odd number)
    face_similarity_threshold: float = 0.4