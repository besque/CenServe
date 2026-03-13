import re
import cv2
import numpy as np
import os, sys
import threading
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from typing import List
from shared_types import DetectionEvent


# ── PII regex patterns ────────────────────────────────────────────────────────

# Patterns that work on the RAW OCR text (may contain spaces, punctuation)
_RAW_PATTERNS = [
    re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b'),                            # PAN
    re.compile(r'\b[\w.+-]+@[\w-]+\.\w{2,}\b'),                        # Email (compact)
    re.compile(r'\b[A-Z]{4}0[A-Z0-9]{6}\b'),                          # IFSC
    re.compile(r'(?i)(password|passwd|pwd)\s*[:\-=]\s*\S+'),           # Visible password
]
_EMAIL_PATTERN = _RAW_PATTERNS[1]

# Patterns that work on DIGITS-ONLY (all non-digit chars stripped)
_DIGIT_PATTERNS = [
    re.compile(r'\d{12}'),              # Aadhaar (12 digits)
    re.compile(r'[6-9]\d{9}'),          # Indian mobile (10 digits starting 6-9)
    re.compile(r'\d{13,19}'),           # Credit/debit card (13-19 digits)
    re.compile(r'\d{6}'),              # OTP / PIN code (6 digits)
]


def _is_pii(text: str) -> bool:
    """Check both raw and digit-normalized forms of the OCR text."""
    raw = text.strip()

    # 1) Email / password / PAN / IFSC etc on the raw string
    if any(p.search(raw) for p in _RAW_PATTERNS):
        return True

    # 2) Email where OCR inserted spaces or slightly mangled punctuation
    #    e.g. "gaganrh7 @ gmail . com" → "gaganrh7@gmail.com"
    #         "gaganrh@gmailcom"       → accept as email-ish
    compact = re.sub(r'\s+', '', raw)
    if '@' in compact:
        # First try the normal email regex on the compact form
        if _EMAIL_PATTERN.search(compact):
            return True
        # Fallback: missing dot before TLD, but still looks like user@domainThing
        user, _, domain = compact.partition('@')
        if len(user) >= 1 and len(domain) >= 3:
            return True

    # 3) Digit-only matching for Aadhaar, cards, phone, OTP
    digits_only = re.sub(r'\D', '', raw)
    if len(digits_only) >= 6 and any(p.search(digits_only) for p in _DIGIT_PATTERNS):
        return True

    return False


# ── Detector class ────────────────────────────────────────────────────────────

class TextPIIDetector:
    def __init__(self, backend: str = 'easy', mode: str = 'camera'):
        """
        mode: 'camera' or 'screen'.
        - camera: moderate resolution
        - screen: higher resolution + central crop for sharper UI text
        """
        self.backend = backend
        self.mode = mode

        if backend == 'easy':
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        else:
            raise ValueError(f'Unknown backend: {backend}')

        print(f'[TextPII] Ready ({backend})')

    def _run_ocr(self, small: np.ndarray):
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        results = []
        raw = self.reader.readtext(rgb)
        for (pts, text, conf) in raw:
            if isinstance(text, str):
                text = text.strip()
            results.append((pts, text, conf))

        return results

    def detect(self, frame: np.ndarray, frame_id: int) -> List[DetectionEvent]:
        h, w = frame.shape[:2]

        # For screen-share we keep text sharper:
        # - crop to central band to remove noisy top/bottom UI chrome
        # - use higher target resolution
        crop_top = 0
        if self.mode == 'screen':
            crop_top = int(h * 0.15)
            crop_bot = int(h * 0.85)
            roi = frame[crop_top:crop_bot, :]
        else:
            roi = frame
            crop_bot = h

        rh, rw = roi.shape[:2]
        target = 960 if self.mode == 'screen' else 640
        scale = target / max(rh, rw)
        if scale < 1.0:
            small = cv2.resize(roi, (int(rw * scale), int(rh * scale)))
        else:
            small = roi
            scale = 1.0
        inv = 1.0 / scale
        CONF_THRESHOLD = 0.35

        events = []
        try:
            t0 = time.time()
            ocr_results = self._run_ocr(small)
            elapsed = time.time() - t0

            for (pts, text, conf) in ocr_results:
                # Log every OCR line so we can see what EasyOCR is actually reading.
                # This helps tune regexes and thresholds.
                print(f'[TextPII] OCR  conf={conf:.2f} text="{text}"')

                if conf < CONF_THRESHOLD:
                    continue

                is_pii = _is_pii(text)
                if is_pii:
                    print(f'[TextPII] HIT  conf={conf:.2f} text="{text}"')
                if not is_pii:
                    continue

                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x1 = int(min(xs) * inv) - 8
                y1 = int(min(ys) * inv) - 8
                x2 = int(max(xs) * inv) + 8
                y2 = int(max(ys) * inv) + 8

                # Map back into full-frame coordinates
                y1 += crop_top
                y2 += crop_top

                x1 = max(0, x1);  y1 = max(0, y1)
                x2 = min(w, x2);  y2 = min(h, y2)

                events.append(DetectionEvent(
                    type='text_pii',
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    frame_id=frame_id,
                    blur=True,
                ))

            if events:
                print(f'[TextPII] Frame {frame_id}: {len(events)} PII region(s) in {elapsed*1000:.0f}ms')

        except Exception as e:
            print(f'[TextPII] OCR error: {e}')

        return events


def make_text_pii_detector(backend: str = 'easy', mode: str = 'camera') -> callable:
    d = TextPIIDetector(backend=backend, mode=mode)
    return d.detect


# ── Async worker ──────────────────────────────────────────────────────────────

class TextPIIWorker:
    """
    Runs TextPIIDetector on a background thread.

    The main loop calls submit_frame() which never blocks.
    The main loop reads latest_events to get cached results.
    """

    def __init__(self, backend: str = 'easy', mode: str = 'camera'):
        self._detector = TextPIIDetector(backend=backend, mode=mode)
        self._lock = threading.Lock()
        self._pending_frame = None
        self._pending_fid = 0
        self._events: List[DetectionEvent] = []
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit_frame(self, frame: np.ndarray, frame_id: int):
        """Drop-in a frame for OCR. Non-blocking; overwrites any older pending frame."""
        with self._lock:
            self._pending_frame = frame.copy()
            self._pending_fid = frame_id

    @property
    def latest_events(self) -> List[DetectionEvent]:
        with self._lock:
            return list(self._events)

    def stop(self):
        self._stop = True

    def _loop(self):
        while not self._stop:
            with self._lock:
                frame = self._pending_frame
                fid = self._pending_fid
                self._pending_frame = None

            if frame is None:
                time.sleep(0.05)
                continue

            try:
                results = self._detector.detect(frame, fid)
                with self._lock:
                    self._events = results
            except Exception as e:
                print(f'[TextPII Worker] error: {e}')

            time.sleep(0.02)
