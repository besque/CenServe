import re
import cv2
import numpy as np
import os, sys
import threading
import time
from collections import deque

os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from typing import List, Tuple
from shared_types import DetectionEvent


# ── PII regex patterns ────────────────────────────────────────────────────────

_RAW_PATTERNS = [
    re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b'),                            # PAN
    re.compile(r'[\w.+-]+@[\w-]+\.[\w]{2,}'),                          # Email (strict)
    re.compile(r'\b[A-Z]{4}0[A-Z0-9]{6}\b'),                          # IFSC
    re.compile(r'(?i)(password|passwd|pwd)\s*[:\-=]\s*\S+'),           # Visible password
]

_DIGIT_PATTERNS = [
    re.compile(r'\d{12}'),              # Aadhaar
    re.compile(r'[6-9]\d{9}'),          # Indian mobile
    re.compile(r'\d{13,19}'),           # Credit/debit card
    re.compile(r'\d{6}'),              # OTP / PIN
]


def _is_pii(text: str) -> bool:
    """Ultra-permissive PII check. Errs on the side of blurring."""
    raw = text.strip()
    if not raw:
        return False

    if any(p.search(raw) for p in _RAW_PATTERNS):
        return True

    compact = re.sub(r'\s+', '', raw)

    # Any token with @ that has chars on both sides is treated as email
    if '@' in compact:
        user, _, domain = compact.partition('@')
        if len(user) >= 1 and len(domain) >= 2:
            return True

    digits_only = re.sub(r'\D', '', raw)
    if len(digits_only) >= 6 and any(p.search(digits_only) for p in _DIGIT_PATTERNS):
        return True

    # Phone-like: +91, 0-prefixed 10-digit, etc.
    phone_compact = re.sub(r'[\s\-\(\)]', '', raw)
    if re.search(r'(\+\d{1,3})?\d{10,}', phone_compact):
        return True

    return False


def _merge_adjacent_tokens(
    ocr_results: list,
    x_gap_px: int = 30,
    y_overlap_ratio: float = 0.5,
) -> list:
    """
    Merge OCR tokens that sit on the same line and are close together.
    This recovers emails/phones that OCR splits across boxes, e.g.
    ["gaganrh@gmail", ".com"] or ["+91", "98765", "43210"].
    Returns a new list with the same (pts, text, conf) format.
    """
    if len(ocr_results) <= 1:
        return ocr_results

    def _bbox(pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    items = []
    for (pts, text, conf) in ocr_results:
        x1, y1, x2, y2 = _bbox(pts)
        items.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                      'text': text, 'conf': conf, 'merged': False})

    items.sort(key=lambda t: (t['y1'], t['x1']))

    merged = []
    for i, a in enumerate(items):
        if a['merged']:
            continue
        group_text = a['text']
        gx1, gy1, gx2, gy2 = a['x1'], a['y1'], a['x2'], a['y2']
        gconf = a['conf']

        for j in range(i + 1, len(items)):
            b = items[j]
            if b['merged']:
                continue
            h_a = gy2 - gy1
            h_b = b['y2'] - b['y1']
            overlap_top = max(gy1, b['y1'])
            overlap_bot = min(gy2, b['y2'])
            overlap = max(0, overlap_bot - overlap_top)
            if overlap < y_overlap_ratio * min(h_a, h_b):
                continue
            gap = b['x1'] - gx2
            if gap > x_gap_px:
                continue
            if gap < -0.5 * min(gx2 - gx1, b['x2'] - b['x1']):
                continue

            group_text = group_text + b['text']
            gx2 = max(gx2, b['x2'])
            gy1 = min(gy1, b['y1'])
            gy2 = max(gy2, b['y2'])
            gconf = min(gconf, b['conf'])
            b['merged'] = True

        pts_merged = [[gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]]
        merged.append((pts_merged, group_text, gconf))

    return merged


# ── Detector class ────────────────────────────────────────────────────────────

class TextPIIDetector:
    def __init__(self, backend: str = 'easy', mode: str = 'camera'):
        """
        backend: 'easy' (EasyOCR) or 'paddle' (PaddleOCR).
        mode: 'camera' or 'screen'.
        """
        self.backend = backend
        self.mode = mode
        self.reader = None

        if backend == 'paddle':
            try:
                from paddleocr import PaddleOCR
                self.reader = PaddleOCR(
                    use_angle_cls=False, lang='en',
                    show_log=False, use_gpu=False,
                )
                print('[TextPII] PaddleOCR loaded')
            except Exception as e:
                print(f'[TextPII] PaddleOCR failed ({e}), falling back to EasyOCR')
                backend = 'easy'
                self.backend = 'easy'

        if backend == 'easy':
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print('[TextPII] EasyOCR loaded')

    def _run_ocr(self, img: np.ndarray) -> list:
        """Returns list of (pts, text, conf). pts = 4 corner points."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.backend == 'paddle':
            return self._ocr_paddle(rgb)
        return self._ocr_easy(rgb)

    def _ocr_easy(self, rgb: np.ndarray) -> list:
        results = []
        for (pts, text, conf) in self.reader.readtext(rgb):
            if isinstance(text, str):
                text = text.strip()
            results.append((pts, text, conf))
        return results

    def _ocr_paddle(self, rgb: np.ndarray) -> list:
        result = self.reader.ocr(rgb, cls=False)
        out = []
        if not result or not result[0]:
            return out
        for line in result[0]:
            # v2 format: [pts, (text, conf)]
            pts = line[0]
            text_conf = line[1]
            text = str(text_conf[0]).strip()
            conf = float(text_conf[1])
            out.append((pts, text, conf))
        return out

    def detect(self, frame: np.ndarray, frame_id: int) -> List[DetectionEvent]:
        h, w = frame.shape[:2]

        crop_top = 0
        if self.mode == 'screen':
            crop_top = int(h * 0.10)
            crop_bot = int(h * 0.90)
            roi = frame[crop_top:crop_bot, :]
        else:
            roi = frame
            crop_bot = h

        rh, rw = roi.shape[:2]
        target = 1280 if self.mode == 'screen' else 640
        scale = target / max(rh, rw)
        if scale < 1.0:
            small = cv2.resize(roi, (int(rw * scale), int(rh * scale)))
        else:
            small = roi
            scale = 1.0
        inv = 1.0 / scale
        CONF_THRESHOLD = 0.30

        events = []
        try:
            t0 = time.time()
            ocr_results = self._run_ocr(small)
            elapsed = time.time() - t0

            # Merge adjacent tokens so split emails/phones get concatenated
            merged_results = _merge_adjacent_tokens(ocr_results, x_gap_px=35)
            all_results = ocr_results + merged_results

            seen_texts = set()
            for (pts, text, conf) in all_results:
                print(f'[TextPII] OCR  conf={conf:.2f} text="{text}"')

                if conf < CONF_THRESHOLD:
                    continue

                if not _is_pii(text):
                    continue

                # Deduplicate: don't blur same text region twice
                text_key = text.strip().lower()
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)

                print(f'[TextPII] HIT  conf={conf:.2f} text="{text}"')

                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x1 = int(min(xs) * inv) - 10
                y1 = int(min(ys) * inv) - 10
                x2 = int(max(xs) * inv) + 10
                y2 = int(max(ys) * inv) + 10

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

# Keep results for this many frame_ids (for display-buffer lookup)
_EVENTS_BY_FID_MAX = 90
# When using display delay, process frames in order. Queue holds ~delay*fps so
# the frame we display (oldest in buffer) has been processed by the time we show it.
_PENDING_QUEUE_MAX = 50


class TextPIIWorker:
    """
    Runs TextPIIDetector on a background thread.
    The main loop calls submit_frame() which never blocks.
    Results are available as latest_events (with TTL) or get_events_for_frame(fid)
    when using a display delay so OCR has time to finish for that frame.
    Uses a small in-order queue so that when the server buffers video by AV_DELAY_SECONDS,
    each displayed frame has its OCR result ready (processed in order, stored by frame_id).
    """

    def __init__(self, backend: str = 'easy', mode: str = 'camera'):
        self._detector = TextPIIDetector(backend=backend, mode=mode)
        self._lock = threading.Lock()
        self._pending_queue: deque = deque(maxlen=_PENDING_QUEUE_MAX)  # (frame, fid)
        self._events: List[DetectionEvent] = []
        self._last_result_fid: int = -1
        self._events_by_fid: dict = {}  # frame_id -> list of events (for delayed display)
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit_frame(self, frame: np.ndarray, frame_id: int):
        """Queue a frame for OCR. Non-blocking; if queue full, newest is dropped so we keep oldest (for display delay)."""
        with self._lock:
            if len(self._pending_queue) >= _PENDING_QUEUE_MAX:
                return
            self._pending_queue.append((frame.copy(), frame_id))

    def get_events_for_frame(self, frame_id: int) -> List[DetectionEvent]:
        """Return OCR results for a specific frame_id (for use with display buffer)."""
        with self._lock:
            return list(self._events_by_fid.get(frame_id, []))

    @property
    def latest_events(self) -> List[DetectionEvent]:
        with self._lock:
            return list(self._events)

    @property
    def last_result_fid(self) -> int:
        with self._lock:
            return self._last_result_fid

    def stop(self):
        self._stop = True

    def _loop(self):
        while not self._stop:
            with self._lock:
                if not self._pending_queue:
                    item = None
                else:
                    item = self._pending_queue.popleft()

            if item is None:
                time.sleep(0.05)
                continue

            frame, fid = item
            try:
                results = self._detector.detect(frame, fid)
                with self._lock:
                    self._events = results
                    self._last_result_fid = fid
                    self._events_by_fid[fid] = results
                    if len(self._events_by_fid) > _EVENTS_BY_FID_MAX:
                        for k in sorted(self._events_by_fid.keys())[:-_EVENTS_BY_FID_MAX]:
                            del self._events_by_fid[k]
            except Exception as e:
                print(f'[TextPII Worker] error: {e}')

            time.sleep(0.02)
