"""
censerve Web Server
Phases: pre_stream → streaming
Run:    python censerve/web/server.py
Open:   http://localhost:5000
"""

import sys, os, time, threading, pickle, cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from flask import Flask, Response, jsonify, send_from_directory, request
from shared_types import DETECTION_CADENCE, CACHE_TTL_FRAMES

app    = Flask(__name__)
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
ENROLLED_DIR  = os.path.join(ROOT, 'enrolled_faces')
ENROLLED_FILE = os.path.join(ENROLLED_DIR, 'faces.pkl')
os.makedirs(ENROLLED_DIR, exist_ok=True)

from censerve.video.screen_capture import ScreenCapture, list_sources as _list_screen_sources

# ── Constants ─────────────────────────────────────────────────────────────
MAX_ENROLLED  = 5
ENROLL_FRAMES = 30

# ── Shared state ──────────────────────────────────────────────────────────
_lock    = threading.Lock()
_jpeg    = None
_running = False

_blur_strength = 55  # must always be odd — OpenCV GaussianBlur kernel size

_state = {
    'phase': 'idle',
}

_settings = {
    'faces':    True,
    'plates':   True,
    'cards':    True,
    'nsfw':     True,
    # Text PII is enabled by default, but only used in screen-share mode
    'text_pii': True,
}

_cap          = None
_face_app     = None
_enrolled_faces = {}  # name → [embedding, ...]

_source_mode    = 'camera'
_screen_capture = None

_evt_begin_stream = threading.Event()

_enroll_state = {
    'active':    False,
    'name':      '',
    'collected': [],
    'progress':  0,
    'msg':       '',
    'done':      False,
}
_enroll_lock = threading.Lock()

# ── Background worker ─────────────────────────────────────────────────────

class _BackgroundWorker:
    """
    Generic async wrapper: runs a detect(frame, frame_id)->List[DetectionEvent]
    callable on a daemon thread so the main loop never blocks.
    """

    def __init__(self, detect_fn, name: str = 'detector'):
        self._detect = detect_fn
        self._name = name
        self._lock = threading.Lock()
        self._pending_frame = None
        self._pending_fid = 0
        self._events = []
        self._last_fid = -1
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit_frame(self, frame, frame_id: int):
        with self._lock:
            self._pending_frame = frame.copy()
            self._pending_fid = frame_id

    @property
    def latest(self):
        """Returns (events, frame_id_of_those_events)."""
        with self._lock:
            return list(self._events), self._last_fid

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
                results = self._detect(frame, fid)
                with self._lock:
                    self._events = results
                    self._last_fid = fid
            except Exception as e:
                print(f'[{self._name}] bg-worker error: {e}')

            time.sleep(0.02)

# ── Helpers ───────────────────────────────────────────────────────────────

def _blur_region(frame, x1, y1, x2, y2, strength=55):
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    k = strength if strength % 2 == 1 else strength + 1
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (k, k), 0)

def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

def _is_owner(embed):
    for embeds in _enrolled_faces.values():
        for e in embeds:
            if _cosine(embed, e) > 0.38:
                return True
    return False

def _encode_jpeg(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes()

# ── Enrolled faces persistence ────────────────────────────────────────────

def _save_enrolled_faces():
    with open(ENROLLED_FILE, 'wb') as f:
        pickle.dump(_enrolled_faces, f)

def _load_enrolled_faces():
    global _enrolled_faces
    if os.path.exists(ENROLLED_FILE):
        with open(ENROLLED_FILE, 'rb') as f:
            _enrolled_faces = pickle.load(f)
    else:
        _enrolled_faces = {}

# ── Enrollment frame collector ────────────────────────────────────────────

def _collect_enrollment_frame(frame):
    with _enroll_lock:
        if not _enroll_state['active']:
            return
        name = _enroll_state['name']

    small = cv2.resize(frame, (640, 360))
    faces = None
    if _face_app:
        try:
            faces = _face_app.get(small)
        except Exception:
            pass

    with _enroll_lock:
        if not _enroll_state['active']:
            return
        if not faces:
            _enroll_state['msg'] = 'No face \u2014 move closer'
            return
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        _enroll_state['collected'].append(f.normed_embedding.copy())
        count = len(_enroll_state['collected'])
        _enroll_state['progress'] = int(count / ENROLL_FRAMES * 100)
        _enroll_state['msg'] = f'{count} / {ENROLL_FRAMES}'
        if count >= ENROLL_FRAMES:
            _enrolled_faces[name] = list(_enroll_state['collected'])
            _enroll_state['active'] = False
            _enroll_state['done'] = True
            _enroll_state['msg'] = f'{name} enrolled'
            _save_enrolled_faces()

# ── Init ──────────────────────────────────────────────────────────────────

def _init_camera():
    global _cap
    _cap = cv2.VideoCapture(0)
    _cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    _cap.set(cv2.CAP_PROP_FPS, 30)
    return _cap.isOpened()

def _init_face_app():
    global _face_app
    try:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
        _face_app.prepare(ctx_id=0, det_size=(320, 320))
        print('[censerve] insightface (buffalo_sc) OK')
    except Exception as e:
        print(f'[censerve] insightface unavailable: {e}')

# ── Pre-stream phase ─────────────────────────────────────────────────────

def _pre_stream_thread():
    global _jpeg

    _evt_begin_stream.clear()
    print('[censerve] Pre-stream \u2014 enroll faces, then click Start Stream')

    while _running and not _evt_begin_stream.is_set():
        ok, frame = _cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        output = frame.copy()

        if _face_app:
            try:
                for f in _face_app.get(frame):
                    if not _is_owner(f.normed_embedding):
                        x1, y1, x2, y2 = [int(v) for v in f.bbox]
                        _blur_region(output, x1-20, y1-20, x2+20, y2+20, _blur_strength)
            except Exception:
                pass

        _collect_enrollment_frame(frame)

        with _lock:
            _jpeg = _encode_jpeg(output)

        time.sleep(0.01)

# ── Streaming ─────────────────────────────────────────────────────────────

def _streaming_thread():
    global _jpeg, _running

    with _lock:
        _state['phase'] = 'streaming'

    # ── Virtual camera ────────────────────────────────────────────────────
    vcam = None
    try:
        import pyvirtualcam
        vcam = pyvirtualcam.Camera(width=1280, height=720, fps=30, backend='obs')
        print('[censerve] OBS virtual camera started')
    except Exception as e:
        print(f'[censerve] Virtual camera skipped: {e}')

    # ── Object detector (plates + cards) — async background worker ────────
    obj_worker = None
    try:
        from censerve.video.object_detector import PlateCardDetector
        _obj = PlateCardDetector()
        obj_worker = _BackgroundWorker(_obj.detect, name='ObjDet')
        print('[censerve] Object detector loaded (async)')
    except Exception as e:
        print(f'[censerve] Object detector skipped: {e}')

    # ── NSFW detector — async background worker ──────────────────────────
    nsfw_worker = None
    try:
        from censerve.video.nsfw_detector import NSFWDetector
        _nsfw = NSFWDetector()
        nsfw_worker = _BackgroundWorker(_nsfw.detect, name='NSFW')
        print('[censerve] NSFW detector loaded (async)')
    except ImportError as e:
        print(f'[censerve] NSFW detector import failed: {e}')
    except Exception as e:
        print(f'[censerve] NSFW detector initialization failed: {e}')
        import traceback
        traceback.print_exc()

    # ── Text PII — uses its own TextPIIWorker ────────────────────────────
    text_pii_worker = None

    # ── Mediapipe fallback for faces ─────────────────────────────────────
    mp_face = None
    if not _face_app:
        try:
            import mediapipe as mp
            mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5)
        except Exception:
            pass

    frame_id = 0
    print('[censerve] Streaming...')

    while _running:
        if _source_mode == 'screen' and _screen_capture:
            ok, frame = _screen_capture.read()
        else:
            ok, frame = _cap.read()

        if not ok or frame is None:
            time.sleep(0.01)
            continue

        with _lock:
            s = dict(_settings)

        is_screen = (_source_mode == 'screen')
        output = frame.copy()

        # ── Face blur (camera only) ──────────────────────────────────────
        if not is_screen and s['faces']:
            if _face_app:
                try:
                    for f in _face_app.get(frame):
                        if not _is_owner(f.normed_embedding):
                            x1, y1, x2, y2 = [int(v) for v in f.bbox]
                            _blur_region(output, x1-20, y1-20, x2+20, y2+20, _blur_strength)
                except Exception:
                    pass
            elif mp_face:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = mp_face.process(rgb)
                    if res.detections:
                        ih, iw = frame.shape[:2]
                        for det in res.detections:
                            bb = det.location_data.relative_bounding_box
                            _blur_region(output,
                                int(bb.xmin*iw)-20, int(bb.ymin*ih)-20,
                                int((bb.xmin+bb.width)*iw)+20,
                                int((bb.ymin+bb.height)*ih)+20, _blur_strength)
                except Exception:
                    pass

        _collect_enrollment_frame(frame)

        # ── Submit frames to async workers at their cadence ──────────────
        obj_cadence = DETECTION_CADENCE['objects_screen' if is_screen else 'objects_camera']
        if obj_worker and frame_id % obj_cadence == 0:
            obj_worker.submit_frame(frame, frame_id)

        nsfw_cadence = DETECTION_CADENCE['nsfw']
        if nsfw_worker and frame_id % nsfw_cadence == 0:
            nsfw_worker.submit_frame(frame, frame_id)

        text_cadence = DETECTION_CADENCE['text_pii_screen' if is_screen else 'text_pii_camera']
        if s.get('text_pii'):
            if text_pii_worker is None:
                try:
                    from censerve.video.text_pii_detector import TextPIIWorker
                    mode = 'screen' if is_screen else 'camera'
                    text_pii_worker = TextPIIWorker(backend='easy', mode=mode)
                    print(f'[censerve] Text PII worker started (mode={mode})')
                except Exception as e:
                    print(f'[censerve] Text PII worker skipped: {e}')
            if text_pii_worker and frame_id % text_cadence == 0:
                text_pii_worker.submit_frame(frame, frame_id)

        # ── Read cached results + TTL expiry ─────────────────────────────
        cached_objs = []
        if obj_worker:
            evts, fid = obj_worker.latest
            if evts and (frame_id - fid) <= CACHE_TTL_FRAMES:
                cached_objs = evts

        cached_nsfw = []
        if nsfw_worker:
            evts, fid = nsfw_worker.latest
            if evts and (frame_id - fid) <= CACHE_TTL_FRAMES:
                cached_nsfw = evts

        cached_text = []
        if text_pii_worker and s.get('text_pii'):
            evts = text_pii_worker.latest_events
            fid = text_pii_worker.last_result_fid
            if evts and fid >= 0 and (frame_id - fid) <= CACHE_TTL_FRAMES:
                cached_text = evts

        # ── Apply blur from all detectors ────────────────────────────────
        for ev in cached_objs:
            if ev.type == 'plate' and s['plates']:
                _blur_region(output, *ev.bbox, _blur_strength)
            elif ev.type == 'card' and s['cards']:
                _blur_region(output, *ev.bbox, _blur_strength)

        if s['nsfw']:
            for ev in cached_nsfw:
                _blur_region(output, *ev.bbox, _blur_strength)

        if s.get('text_pii'):
            for ev in cached_text:
                _blur_region(output, *ev.bbox, _blur_strength)

        # ── Output ───────────────────────────────────────────────────────
        with _lock:
            _jpeg = _encode_jpeg(output)

        if vcam:
            try:
                rgb_out = cv2.cvtColor(cv2.resize(output, (1280, 720)), cv2.COLOR_BGR2RGB)
                vcam.send(rgb_out)
                vcam.sleep_until_next_frame()
            except Exception:
                pass

        frame_id += 1

    # ── Cleanup ──────────────────────────────────────────────────────────
    if obj_worker:
        obj_worker.stop()
    if nsfw_worker:
        nsfw_worker.stop()
    if text_pii_worker:
        text_pii_worker.stop()
    if vcam:
        vcam.close()
    with _lock:
        _state['phase'] = 'idle'
    print('[censerve] Stopped.')

# ── Entry ─────────────────────────────────────────────────────────────────

def _start_thread():
    global _running, _screen_capture, _jpeg
    _running = True
    if not _init_camera():
        print('[censerve] ERROR: cannot open webcam')
        _running = False
        return
    _screen_capture = ScreenCapture()
    with _lock:
        _state['phase'] = 'starting'
    ok, frame = _cap.read()
    if ok:
        with _lock:
            _jpeg = _encode_jpeg(frame)
    _init_face_app()
    _load_enrolled_faces()
    with _lock:
        _state['phase'] = 'pre_stream'
    _pre_stream_thread()
    _streaming_thread()

# ── Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    r = send_from_directory(STATIC, 'index.html')
    r.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    r.headers['Pragma'] = 'no-cache'
    return r

@app.route('/start', methods=['POST'])
def start():
    global _running
    with _lock:
        already = _running
    if not already:
        threading.Thread(target=_start_thread, daemon=True).start()
    return jsonify({'ok': True})

@app.route('/blur_strength', methods=['POST'])
def set_blur_strength():
    global _blur_strength
    data = request.get_json() or {}
    val = int(data.get('value', _blur_strength))
    val = max(11, min(99, val))
    if val % 2 == 0:
        val += 1
    _blur_strength = val
    return jsonify({'blur_strength': _blur_strength})

@app.route('/toggle/<feature>', methods=['POST'])
def toggle(feature):
    if feature not in _settings:
        return jsonify({'error': 'unknown'}), 400
    with _lock:
        _settings[feature] = not _settings[feature]
        val = _settings[feature]
    return jsonify({'feature': feature, 'enabled': val})

@app.route('/status')
def get_status():
    with _lock:
        return jsonify({
            'running':  _running,
            'phase':    _state['phase'],
            'settings': dict(_settings),
        })

@app.route('/stop', methods=['POST'])
def stop():
    global _running
    _running = False
    return jsonify({'ok': True})

@app.route('/screens', methods=['GET'])
def get_screens():
    try:
        screens = _list_screen_sources()
    except Exception as e:
        print(f'[Screens] {e}')
        screens = []
    return jsonify({'screens': screens})

@app.route('/source', methods=['POST'])
def set_source():
    global _source_mode
    data = request.get_json() or {}
    mode = data.get('mode', 'camera')
    if mode == 'screen':
        source = data.get('source')
        if not source:
            return jsonify({'error': 'source required for screen mode'}), 400
        if _screen_capture:
            _screen_capture.set_source(source)
        _source_mode = 'screen'
        print(f'[censerve] Source \u2192 screen: {source.get("label")}')
    else:
        _source_mode = 'camera'
        print('[censerve] Source \u2192 camera')
    return jsonify({'ok': True, 'mode': _source_mode})

# ── Face enrollment routes ────────────────────────────────────────────────

@app.route('/faces', methods=['GET'])
def get_faces():
    return jsonify({'faces': list(_enrolled_faces.keys())})

@app.route('/faces/<name>', methods=['DELETE'])
def delete_face(name):
    if name in _enrolled_faces:
        del _enrolled_faces[name]
        _save_enrolled_faces()
    return jsonify({'ok': True})

@app.route('/enroll/start', methods=['POST'])
def enroll_start():
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'name is required'}), 400
    if len(_enrolled_faces) >= MAX_ENROLLED and name not in _enrolled_faces:
        return jsonify({'error': 'Maximum 5 faces allowed'}), 400
    with _enroll_lock:
        _enroll_state['active']    = True
        _enroll_state['name']      = name
        _enroll_state['collected'] = []
        _enroll_state['progress']  = 0
        _enroll_state['msg']       = 'Look at the camera'
        _enroll_state['done']      = False
    return jsonify({'ok': True})

@app.route('/enroll/status', methods=['GET'])
def enroll_status():
    with _enroll_lock:
        return jsonify({
            'active':   _enroll_state['active'],
            'name':     _enroll_state['name'],
            'progress': _enroll_state['progress'],
            'msg':      _enroll_state['msg'],
            'done':     _enroll_state['done'],
        })

@app.route('/enroll/cancel', methods=['POST'])
def enroll_cancel():
    with _enroll_lock:
        _enroll_state['active'] = False
        _enroll_state['msg']    = 'Cancelled'
    return jsonify({'ok': True})

@app.route('/stream/start', methods=['POST'])
def stream_start():
    with _lock:
        phase = _state['phase']
    if phase != 'pre_stream':
        return jsonify({'error': f'Cannot start stream in phase: {phase}'}), 400
    _evt_begin_stream.set()
    return jsonify({'ok': True})

def _gen():
    while True:
        with _lock:
            frame = _jpeg
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

@app.route('/video_feed')
def video_feed():
    r = Response(_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
    r.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    r.headers['Pragma'] = 'no-cache'
    return r

if __name__ == '__main__':
    print('censerve \u2192 http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, threaded=True)
