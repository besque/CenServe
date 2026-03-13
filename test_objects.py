import sys
sys.path.insert(0, ".")
import cv2
from shared_types import PipelineConfig
from censerve.video.face_pipeline import FacePipeline
from censerve.video.object_detector import PlateCardDetector
from censerve.video.nsfw_detector import NSFWDetector
from censerve.video.blur_compositor import apply_blurs, draw_debug_overlay
from censerve.video.tracker import MultiObjectTracker

config = PipelineConfig()

face_pipeline  = FacePipeline(config)
plate_detector = PlateCardDetector()
nsfw_detector  = NSFWDetector()
tracker        = MultiObjectTracker(max_age=60, min_hits=1, iou_threshold=0.15)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

frame_id         = 0
last_detections  = []   # plates + cards
last_nsfw_events = []   # persisted separately — NudeNet is slow

print("✅ Running. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    # ── Faces: every frame ──────────────────────────────────────────
    try:
        face_events = face_pipeline.detect_faces(frame, frame_id)
    except Exception as e:
        face_events = []

    # ── Plates + Cards: every 3 frames ──────────────────────────────
    if frame_id % 3 == 0:
        try:
            last_detections = plate_detector.detect(frame, frame_id)
            if last_detections:
                print(f"Frame {frame_id}: {[(e.type, round(e.confidence,2)) for e in last_detections]}")
        except Exception as e:
            print(f"❌ Object detect error: {e}")

    # ── NSFW: every 30 frames (NudeNet is slow ~200ms) ──────────────
    if frame_id % 30 == 0:
        try:
            last_nsfw_events = nsfw_detector.detect(frame, frame_id)
            if last_nsfw_events:
                print(f"Frame {frame_id}: NSFW {[(e.type, round(e.confidence,2)) for e in last_nsfw_events]}")
        except Exception as e:
            print(f"❌ NSFW error: {e}")

    # ── Tracker: feed detections, returns smooth boxes ───────────────
    # update() handles both prediction AND matching in one call
    tracked = tracker.update(last_detections)

    # ── Combine all events ───────────────────────────────────────────
    all_events = face_events + tracked + last_nsfw_events

    # ── Blur + display ───────────────────────────────────────────────
    output = apply_blurs(frame, all_events, config)
    output = draw_debug_overlay(output, all_events)

    cv2.imshow("censerve Test", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()