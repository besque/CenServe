"""
censerve Face Recognition + Virtual Camera

Face enrollment and recognition with virtual camera output for Zoom/Discord integration.
"""
import cv2
import numpy as np
import sys
import time
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared_types import PipelineConfig
from censerve.video.virtual_video_loop import VirtualVideoLoop
from censerve.video.face_pipeline import FacePipeline

# Virtual camera import
try:
    import pyvirtualcam
    VIRTUAL_CAMERA_AVAILABLE = True
    print("✅ Virtual camera available")
except ImportError:
    VIRTUAL_CAMERA_AVAILABLE = False
    print("⚠️  Virtual camera not available, using local display")

FRAMES_NEEDED = 60   # collect 60 frames for enrollment

def test_face_recognition():
    """Test face recognition with virtual camera output"""
    print("\n=== censerve Face Recognition + Virtual Camera ===")
    print("Press E to enroll your face")
    print("Press S to skip enrollment and start virtual camera stream")
    print("Press Q to quit\n")

    # Configuration
    config = PipelineConfig(
        blur_faces=True,
        detection_cadence=15,
        blur_strength=51,
        face_similarity_threshold=0.38
    )

    # Initialize face pipeline
    face_pipeline = FacePipeline(config)

    # Setup camera for enrollment
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    collecting = False
    embeddings_collected = []
    enrolled = False

    # Enrollment phase
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        faces = face_pipeline.face_app.get(frame)

        if collecting:
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                embeddings_collected.append(face.embedding)

                # Draw progress bar
                progress = len(embeddings_collected) / FRAMES_NEEDED
                bar_w = int(400 * progress)
                cv2.rectangle(display, (120, 680), (520, 700), (50, 50, 50), -1)
                cv2.rectangle(display, (120, 680), (120 + bar_w, 700), (0, 220, 100), -1)
                cv2.putText(display, f"Collecting: {len(embeddings_collected)}/{FRAMES_NEEDED}",
                            (120, 675), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 100), 2)

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
        else:
            # Draw face boxes while idle
            for face in faces:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                cv2.rectangle(display, (x1, y1), (x2, y2), (200, 200, 200), 1)

        # Status overlay
        status = ""
        if enrolled:
            status = "Face enrolled! Press S to start virtual camera stream"
            color = (0, 220, 100)
        elif collecting:
            status = "Collecting frames..."
            color = (0, 200, 255)
        else:
            status = "Press E to enroll your face | Press S to skip enrollment"
            color = (200, 200, 200)

        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Virtual camera status
        if VIRTUAL_CAMERA_AVAILABLE:
            cv2.putText(display, "🎥 Virtual Camera Ready", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("censerve — Face Enrollment", display)

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
            break

    # ── Start virtual camera streaming phase ──
    print("\n[censerve] Starting virtual camera stream...")
    if enrolled:
        print("Your enrolled face will NOT be blurred. All others will be.")
    else:
        print("No face enrolled. All detected faces will be blurred.")
    print("🎥 Stream is available in Zoom/Discord as 'OBS Virtual Camera'")
    print("Press Q to quit.\n")

    # Create virtual video loop
    loop = VirtualVideoLoop(config, source=0, debug=False)

    # Pass the already-enrolled face pipeline into the loop
    loop.face_pipeline = face_pipeline

    try:
        # Start virtual camera stream
        loop.run(use_virtual_camera=True)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        loop.stop()
        print("Virtual camera stream stopped")

if __name__ == "__main__":
    print("🎭 censerve AI - Face Recognition Virtual Camera")
    print("=" * 60)
    
    # Check virtual camera availability
    if not VIRTUAL_CAMERA_AVAILABLE:
        print("⚠️  Warning: Virtual camera not available")
        print("   Install with: pip install pyvirtualcam")
        print("   Or install OBS Studio for virtual camera driver")
        response = input("Continue with local display? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(0)
    
    # Run face recognition test
    test_face_recognition()
