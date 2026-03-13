"""
Text PII Detector — finds sensitive text (card numbers, Aadhaar, PAN) in video frames
using OCR and regex pattern matching.
"""

import re, cv2
from shared_types import DetectionEvent

_PATTERNS = [
    re.compile(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'),  # 16-digit card
    re.compile(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'),               # 12-digit Aadhaar
    re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b'),                           # PAN
]

def _is_pii(text):
    clean = text.strip()
    for pat in _PATTERNS:
        if pat.search(clean):
            return True
    return False


def make_text_pii_detector(backend='easy'):
    """
    Returns a callable  detect(frame, frame_id) -> list[DetectionEvent].
    backend='easy' uses EasyOCR.
    """
    if backend == 'easy':
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    else:
        raise ValueError(f'Unknown backend: {backend}')

    def detect(frame, frame_id):
        results = []
        small = cv2.resize(frame, (640, 360))
        sx = frame.shape[1] / 640
        sy = frame.shape[0] / 360

        ocr_results = reader.readtext(small)
        for (bbox_pts, text, conf) in ocr_results:
            if not _is_pii(text):
                continue
            xs = [int(p[0] * sx) for p in bbox_pts]
            ys = [int(p[1] * sy) for p in bbox_pts]
            x1, y1 = min(xs) - 10, min(ys) - 10
            x2, y2 = max(xs) + 10, max(ys) + 10
            results.append(DetectionEvent(
                type='text_pii',
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                frame_id=frame_id,
            ))
        return results

    return detect
