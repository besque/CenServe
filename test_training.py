import cv2, sys, time
sys.path.insert(0, ".")
from shared_types import PipelineConfig
from censerve.video.tracker import MultiObjectTracker
from censerve.video.object_detector import PlateCardDetector
from censerve.video.blur_compositor import apply_blurs

config  = PipelineConfig()
tracker = MultiObjectTracker(max_age=45, min_hits=1, iou_threshold=0.25)
det     = PlateCardDetector()
cap     = cv2.VideoCapture(0)
frame_id = 0

print("Hold a license plate or card to camera. Move it around. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_id += 1

    # Simulate T3 firing every 5 frames
    if frame_id % 5 == 0:
        detections = det.detect(frame, frame_id)
        tracker.update(detections)

    # T4 runs every frame
    boxes = tracker.update([])

    out = apply_blurs(frame, boxes, config)

    # Debug: draw raw tracker boxes in green
    for b in boxes:
        x1,y1,x2,y2 = b.bbox
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.putText(out, b.type, (x1, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

    cv2.imshow("Tracking Test", out)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()