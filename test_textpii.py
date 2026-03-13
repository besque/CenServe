"""
Test script for TextPII detection.
Uses the async TextPIIWorker so the video feed stays smooth.
Shows blurred PII regions with green outlines in a live webcam window.

Usage:
    python test_textpii.py          # webcam
    python test_textpii.py screen   # capture primary monitor instead
"""
import sys
import cv2
import time

sys.path.insert(0, ".")
from censerve.video.text_pii_detector import TextPIIWorker

USE_SCREEN = len(sys.argv) > 1 and sys.argv[1] == 'screen'

if USE_SCREEN:
    from censerve.video.screen_capture import ScreenCapture
    cap = ScreenCapture()
    cap.set_source({'type': 'monitor', 'index': 1})
    print('[Test] Capturing primary monitor')
else:
    cap = cv2.VideoCapture(0)
    print('[Test] Capturing webcam')

mode = 'screen' if USE_SCREEN else 'camera'
worker = TextPIIWorker(backend='easy', mode=mode)

frame_id = 0
SUBMIT_EVERY = 8 if USE_SCREEN else 20

print('[Test] Running. Press Q to quit.')

while True:
    if USE_SCREEN:
        ok, frame = cap.read()
    else:
        ok, frame = cap.read()

    if not ok or frame is None:
        time.sleep(0.01)
        continue

    if frame_id % SUBMIT_EVERY == 0:
        worker.submit_frame(frame, frame_id)

    events = worker.latest_events

    for ev in events:
        x1, y1, x2, y2 = ev.bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (55, 55), 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
            cv2.putText(frame, 'PII', (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)

    status = f'Frame {frame_id} | PII regions: {len(events)}'
    cv2.putText(frame, status, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('TextPII Test', frame)
    frame_id += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

worker.stop()
if USE_SCREEN:
    cap.close()
else:
    cap.release()
cv2.destroyAllWindows()
