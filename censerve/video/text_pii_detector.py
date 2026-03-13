import re
import cv2
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from typing import List
from shared_types import DetectionEvent


# ── PII regex patterns ────────────────────────────────────────────────────────

PII_PATTERNS = [
    re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b'),                         # Aadhaar
    re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b'),                            # PAN
    re.compile(r'\b[6-9]\d{9}\b'),                                     # Indian mobile
    re.compile(r'\b[\w.+-]+@[\w-]+\.\w{2,}\b'),                        # Email
    re.compile(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'),    # Credit card
    re.compile(r'\b[A-Z]{4}0[A-Z0-9]{6}\b'),                          # IFSC
    re.compile(r'\b\d{6}\b'),                                          # OTP
    re.compile(r'(?i)(password|passwd|pwd)\s*[:\-=]\s*\S+'),           # Visible password
]

def _is_pii(text: str) -> bool:
    return any(p.search(text) for p in PII_PATTERNS)

# ── Detector class ────────────────────────────────────────────────────────────

class TextPIIDetector:
    def __init__(self, backend: str = 'easy'):
        self.backend = backend

        if backend == 'easy':
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            pass

        else:
            raise ValueError(f'Unknown backend: {backend}')

        print(f'[TextPII] Ready ({backend})')

    def _run_ocr(self, small: np.ndarray):
        results = []

        raw = self.reader.readtext(small)
        for (pts, text, conf) in raw:
            results.append((pts, text, conf))

        return results

    def detect(self, frame: np.ndarray, frame_id: int) -> List[DetectionEvent]:
        """
        Main entry point. Called by the background detection thread.

        Two confidence thresholds:
          - Camera feed: 0.70  (physical documents are noisier)
          - Screen share: 0.50 (digital text is clean and sharp)

        The caller passes the same frame either way — the detector doesn't
        need to know the source. Person 1 will call this with different cadences
        depending on source mode.
        """
        h, w = frame.shape[:2]

        # Downscale to 640px wide for speed — OCR doesn't need full resolution
        scale = 640 / max(h, w)
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame
            scale = 1.0
        inv = 1.0 / scale
        CONF_THRESHOLD = 0.6

        events = []
        try:
            ocr_results = self._run_ocr(small)
            for (pts, text, conf) in ocr_results:
                if conf < CONF_THRESHOLD:
                    continue
                if not _is_pii(text):
                    continue

                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x1 = int(min(xs) * inv) - 6
                y1 = int(min(ys) * inv) - 6
                x2 = int(max(xs) * inv) + 6
                y2 = int(max(ys) * inv) + 6

                # Clamp to frame bounds
                x1 = max(0, x1);  y1 = max(0, y1)
                x2 = min(w, x2);  y2 = min(h, y2)

                events.append(DetectionEvent(
                    type='text_pii',
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    frame_id=frame_id,
                    blur=True,
                ))

        except Exception as e:
            print(f'[TextPII] OCR error: {e}')

        return events


def make_text_pii_detector(backend: str = 'easy') -> callable:
    """
    Factory function. Returns the detect() method ready to register.
    Person 1 imports and calls this.

    Usage in server.py:
        from blurberry.video.text_pii_detector import make_text_pii_detector
        text_det = make_text_pii_detector(backend='easy')
        events = text_det(frame, frame_id)
    """
    d = TextPIIDetector(backend=backend)
    return d.detect