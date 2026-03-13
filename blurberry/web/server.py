"""
BlurBerry Web Server
Phases: ready → collecting (E) → enrolled → streaming (S)
Run:    python blurberry/web/server.py
Open:   http://localhost:5000
"""

import sys, os, time, threading, pickle, cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from flask import Flask, Response, jsonify, send_from_directory

app    = Flask(__name__)
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
ENROLLED_DIR  = os.path.join(ROOT, 'enrolled_faces')
ENROLLED_FILE = os.path.join(ENROLLED_DIR, 'owner.pkl')
os.makedirs(ENROLLED_DIR, exist_ok=True)

# ── Shared state ───────────────────────────────────────────────────────────────
_lock    = threading.Lock()
_jpeg    = None
_running = False

# phases: idle | ready | collecting | enrolled | streaming
_state = {
    'phase':           'idle',
    'enroll_progress': 0,   # 0-100
    'enroll_msg':      '',
}

_settings = {
    'faces':  True,
    'plates': True,
    'cards':  True,
    'nsfw':   True,
}

# Events set by keypress route to unblock waiting threads
_evt_start_collect = threading.Event()
_evt_start_stream  = threading.Event()

_cap          = None
_face_app     = None
_owner_embeds = []

ENROLL_FRAMES = 30

# ── Helpers ────────────────────────────────────────────────────────────────────

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
    if not _owner_embeds:
        return False
    return max(_cosine(embed, e) for e in _owner_embeds) > 0.38

def _encode_jpeg(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes()

# ── Init ───────────────────────────────────────────────────────────────────────

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
        print('[BlurBerry] insightface (buffalo_sc) OK')
    except Exception as e:
        print(f'[BlurBerry] insightface unavailable: {e}')

# ── Phase 1: ready — show live preview, wait for E ────────────────────────────

def _ready_phase():
    with _lock:
        _state['phase'] = 'ready'

    # Produce first frame so /video_feed has something to send as soon as client connects
    ok, frame = _cap.read()
    if ok:
        with _lock:
            _jpeg = _encode_jpeg(frame)

    print('[BlurBerry] Ready — waiting for E to start enrollment')
    _evt_start_collect.clear()

    while not _evt_start_collect.wait(timeout=0.05):
        ok, frame = _cap.read()
        if not ok:
            continue
        with _lock:
            _jpeg = _encode_jpeg(frame)

    _collect_phase()

# ── Phase 2: collecting embeddings ────────────────────────────────────────────

def _collect_phase():
    global _owner_embeds

    with _lock:
        _state['phase']           = 'collecting'
        _state['enroll_progress'] = 0
        _state['enroll_msg']      = 'Look at the camera'

    # Always fresh
    if os.path.exists(ENROLLED_FILE):
        os.remove(ENROLLED_FILE)
    _owner_embeds = []

    collected  = []
    enroll_tick = 0
    print(f'[Enroll] Collecting {ENROLL_FRAMES} embeddings...')

    while len(collected) < ENROLL_FRAMES:
        ok, frame = _cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        display = frame.copy()
        embed   = None

        # Run detection every other frame for speed
        if _face_app and enroll_tick % 2 == 0:
            try:
                small  = cv2.resize(frame, (640, 360))
                faces  = _face_app.get(small)
                if faces:
                    sx = frame.shape[1] / 640
                    sy = frame.shape[0] / 360
                    f  = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                    embed = f.normed_embedding
                    collected.append(embed.copy())
                    x1 = int(f.bbox[0]*sx); y1 = int(f.bbox[1]*sy)
                    x2 = int(f.bbox[2]*sx); y2 = int(f.bbox[3]*sy)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 229, 160), 2)
            except Exception:
                pass
        enroll_tick += 1

        pct = int(len(collected) / ENROLL_FRAMES * 100)
        msg = 'No face detected — move closer' if embed is None else f'{len(collected)}/{ENROLL_FRAMES}'

        with _lock:
            _state['enroll_progress'] = pct
            _state['enroll_msg']      = msg
            _jpeg = _encode_jpeg(display)

    # Save
    with open(ENROLLED_FILE, 'wb') as f:
        pickle.dump(collected, f)
    _owner_embeds = collected
    print(f'[Enroll] Done — {len(collected)} embeddings saved')

    with _lock:
        _state['phase']           = 'enrolled'
        _state['enroll_progress'] = 100
        _state['enroll_msg']      = 'Enrolled!'

    print('[BlurBerry] Enrolled — waiting for S to start stream')
    _evt_start_stream.clear()

    # Keep showing live preview while waiting for S
    while not _evt_start_stream.wait(timeout=0.05):
        ok, frame = _cap.read()
        if not ok:
            continue
        with _lock:
            _jpeg = _encode_jpeg(frame)

    _streaming_thread()

# ── Phase 3: streaming ─────────────────────────────────────────────────────────

def _streaming_thread():
    global _jpeg, _running

    with _lock:
        _state['phase'] = 'streaming'

    vcam = None
    try:
        import pyvirtualcam
        vcam = pyvirtualcam.Camera(width=1280, height=720, fps=30, backend='obs')
        print('[BlurBerry] OBS virtual camera started')
    except Exception as e:
        print(f'[BlurBerry] Virtual camera skipped: {e}')

    obj_det = None
    try:
        from blurberry.video.object_detector import PlateCardDetector
        obj_det = PlateCardDetector()
        print('[BlurBerry] Object detector loaded')
    except Exception as e:
        print(f'[BlurBerry] Object detector skipped: {e}')

    nsfw_detect = None
    try:
        from blurberry.video.nsfw_detector import make_nsfw_detector
        nsfw_detect = make_nsfw_detector()
        print('[BlurBerry] NSFW detector loaded')
    except Exception as e:
        print(f'[BlurBerry] NSFW detector skipped: {e}')

    mp_face = None
    if not _face_app:
        try:
            import mediapipe as mp
            mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5)
        except Exception:
            pass

    frame_id    = 0
    cached_objs = []
    cached_nsfw = []
    print('[BlurBerry] Streaming...')

    while _running:
        ok, frame = _cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        with _lock:
            s = dict(_settings)

        output = frame.copy()

        if s['faces']:
            if _face_app:
                try:
                    for f in _face_app.get(frame):
                        if not _is_owner(f.normed_embedding):
                            x1,y1,x2,y2 = [int(v) for v in f.bbox]
                            _blur_region(output, x1-20, y1-20, x2+20, y2+20, 71)
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
                                int((bb.ymin+bb.height)*ih)+20, 71)
                except Exception:
                    pass

        if frame_id % 10 == 0 and obj_det:
            try:
                cached_objs = obj_det.detect(frame, frame_id)
            except Exception:
                cached_objs = []

        if frame_id % 15 == 0 and nsfw_detect:
            try:
                cached_nsfw = nsfw_detect(frame, frame_id)
            except Exception:
                cached_nsfw = []

        for ev in cached_objs:
            if ev.type == 'plate' and s['plates']:
                _blur_region(output, *ev.bbox)
            elif ev.type == 'card' and s['cards']:
                _blur_region(output, *ev.bbox)

        if s['nsfw']:
            for ev in cached_nsfw:
                _blur_region(output, *ev.bbox)

        with _lock:
            _jpeg = _encode_jpeg(output)

        if vcam:
            try:
                rgb_out = cv2.cvtColor(cv2.resize(output, (1280,720)), cv2.COLOR_BGR2RGB)
                vcam.send(rgb_out)
                vcam.sleep_until_next_frame()
            except Exception:
                pass

        frame_id += 1

    if vcam:
        vcam.close()
    with _lock:
        _state['phase'] = 'idle'
    print('[BlurBerry] Stopped.')

# ── Entry ──────────────────────────────────────────────────────────────────────

def _start_thread():
    global _running
    _running = True
    if not _init_camera():
        print('[BlurBerry] ERROR: cannot open webcam')
        _running = False
        return
    # Show first frame immediately so /video_feed has something while face app loads
    with _lock:
        _state['phase'] = 'starting'
    ok, frame = _cap.read()
    if ok:
        with _lock:
            _jpeg = _encode_jpeg(frame)
    _init_face_app()
    _ready_phase()

# ── Routes ─────────────────────────────────────────────────────────────────────

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

@app.route('/keypress/<key>', methods=['POST'])
def keypress(key):
    k = key.lower()
    with _lock:
        phase = _state['phase']
    if k == 'e' and phase == 'ready':
        _evt_start_collect.set()
        return jsonify({'ok': True, 'action': 'start_collect'})
    if k == 's' and phase == 'enrolled':
        _evt_start_stream.set()
        return jsonify({'ok': True, 'action': 'start_stream'})
    return jsonify({'ok': False, 'reason': f'key {k} not valid in phase {phase}'})

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
            'running':          _running,
            'phase':            _state['phase'],
            'enroll_progress':  _state['enroll_progress'],
            'enroll_msg':       _state['enroll_msg'],
            'settings':         dict(_settings),
        })

@app.route('/stop', methods=['POST'])
def stop():
    global _running
    _running = False
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
    print('BlurBerry → http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, threaded=True)