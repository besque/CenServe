import cv2
import threading
from censerve.video.text_pii_detector import make_text_pii_detector

detect = make_text_pii_detector(backend='easy')

cap = cv2.VideoCapture(0)
frame_id = 0
ocr_running = False
tracked = []
TTL = 60

def run_ocr(frame, fid):
    global ocr_running, tracked
    events = detect(frame, fid)
    new_tracked = []
    for ev in events:
        x1, y1, x2, y2 = ev.bbox
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        new_tracked.append({'bbox': (x1, y1, x2, y2), 'ttl': TTL, 'tracker': tracker})
    if new_tracked:
        tracked = new_tracked
    ocr_running = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not ocr_running:
        ocr_running = True
        t = threading.Thread(target=run_ocr, args=(frame.copy(), frame_id))
        t.daemon = True
        t.start()

    still_alive = []
    for item in tracked:
        ok, box = item['tracker'].update(frame)
        if ok and item['ttl'] > 0:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 100), 2)
            cv2.putText(frame, 'PII', (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)
            item['ttl'] -= 1
            still_alive.append(item)
    tracked = still_alive

    cv2.imshow('TextPII Test', frame)
    frame_id += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()