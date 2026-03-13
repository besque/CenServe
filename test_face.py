"""
censerve — Webcam enrollment 

Controls:
  E  →  Start collecting enrollment frames 
  S  →  Stop collecting, save enrollment, start privacy stream
  Q  →  Quit anytime
"""
import cv2
import numpy as np
import sys
import time
sys.path.insert(0, ".")

from shared_types import PipelineConfig
from censerve.video.face_pipeline import FacePipeline
from censerve.video.video_loop import VideoLoop

FRAMES_NEEDED = 60   # collect 60 frames for enrollment — good balance of speed vs accuracy

config = PipelineConfig(
    blur_faces=True,
    detection_cadence=15,
    blur_strength=51,
    face_similarity_threshold=0.38
)

face_pipeline = FacePipeline(config)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n=== censerve Enrollment ===")
print("Press E to start collecting your face")
print("Press S when done to save and start stream")
print("Press Q to quit\n")

collecting = False
embeddings_collected = []
enrolled = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    faces = face_pipeline.face_app.get(frame)

    if collecting:
        # Try to grab embedding from the largest face
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            embeddings_collected.append(face.embedding)

            # Draw progress bar
            progress = len(embeddings_collected) / FRAMES_NEEDED
            bar_w = int(400 * progress)
            cv2.rectangle(display, (120, 440), (520, 460), (50, 50, 50), -1)
            cv2.rectangle(display, (120, 440), (120 + bar_w, 460), (0, 220, 100), -1)
            cv2.putText(display, f"Collecting: {len(embeddings_collected)}/{FRAMES_NEEDED}",
                        (120, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 100), 2)

            # Draw box around face being enrolled
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 100), 2)
            cv2.putText(display, "Enrolling...", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 100), 2)
        else:
            cv2.putText(display, "No face detected — look at camera",
                        (120, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

        # Auto-finish when enough frames collected
        if len(embeddings_collected) >= FRAMES_NEEDED:
            collecting = False
            face_pipeline.enroll_from_embeddings(embeddings_collected, name="owner")
            enrolled = True
            print(f"[Enrollment] Done! Collected {len(embeddings_collected)} frames.")
            print("Press S to start the privacy stream, or keep collecting more frames.")

    else:
        # Draw face boxes while idle
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            cv2.rectangle(display, (x1, y1), (x2, y2), (200, 200, 200), 1)

    # Status overlay
    status = ""
    if enrolled:
        status = "Face enrolled! Press S to start stream"
        color = (0, 220, 100)
    elif collecting:
        status = "Collecting frames..."
        color = (0, 200, 255)
    else:
        status = "Press E to enroll your face | Press S to skip enrollment"
        color = (200, 200, 200)

    cv2.putText(display, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.imshow("censerve — Enrollment", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    elif key == ord('e') and not collecting:
        embeddings_collected = []
        collecting = True
        print("[Enrollment] Started collecting frames. Hold still and face the camera.")

    elif key == ord('s'):
        cap.release()
        cv2.destroyAllWindows()
        break   # move to stream

# ── Start the actual privacy stream ──
print("\n[censerve] Starting stream...")
print("Your face will NOT be blurred. All others will be.")
print("Press Q to quit.\n")

loop = VideoLoop(config, source=0, debug=False)

# Pass the already-enrolled face pipeline into the loop so we don't re-download models
loop.face_pipeline = face_pipeline

loop.run()