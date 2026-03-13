"""
censerve Complete Virtual Camera Script

This script:
1. Runs camera with face enrollment
2. Starts virtual camera with privacy protection
3. Blocks: License plates, Credit cards, NSFW content, Text PII
4. Whitelists enrolled faces (your face stays clear)
5. Streams to Zoom/Discord via OBS Virtual Camera
"""
import cv2
import numpy as np
import sys
import time
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared_types import PipelineConfig
from censerve.video.virtual_video_loop import VirtualVideoLoop
from censerve.video.face_pipeline import FacePipeline
from censerve.video.object_detector import PlateCardDetector

# Virtual camera import
try:
    import pyvirtualcam
    VIRTUAL_CAMERA_AVAILABLE = True
    print("✅ Virtual camera available")
except ImportError:
    VIRTUAL_CAMERA_AVAILABLE = False
    print("⚠️  Virtual camera not available, using local display")

FRAMES_NEEDED = 60   # collect 60 frames for enrollment

def complete_virtual_camera():
    """Complete virtual camera with face enrollment and privacy protection"""
    print("\n" + "="*60)
    print("🎭 censerve AI - Complete Virtual Camera")
    print("="*60)
    print("\n🔒 Privacy Protection Features:")
    print("   ✅ Face Recognition (your face stays clear)")
    print("   ❌ License Plates (auto-blurred)")
    print("   ❌ Credit Cards (auto-blurred)")
    print("   ❌ NSFW Content (auto-blurred)")
    print("   ❌ Text PII (auto-blurred)")
    print("\n📋 Controls:")
    print("   E - Enroll your face")
    print("   S - Skip enrollment, start virtual camera")
    print("   Q - Quit")
    print("\n🎥 Output: OBS Virtual Camera (for Zoom/Discord)")
    print("="*60)

    # Configuration with all privacy features enabled
    config = PipelineConfig(
        blur_faces=True,           # Enable face blur (whitelisted faces won't be blurred)
        blur_plates=True,          # Blur license plates
        blur_cards=True,           # Blur credit cards
        blur_nsfw=True,            # Blur NSFW content
        blur_text_pii=True,        # Blur text PII
        detection_cadence=10,      # Run object detection every 10 frames
        blur_strength=51,          # Gaussian blur kernel size
        face_similarity_threshold=0.38  # Face recognition threshold
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
                cv2.putText(display, f"Enrolling: {len(embeddings_collected)}/{FRAMES_NEEDED}",
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
                print(f"\n✅ [Enrollment] Done! Collected {len(embeddings_collected)} frames.")
                print("🎯 Your face is now whitelisted and will stay clear!")
        else:
            # Draw face boxes while idle
            for face in faces:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                cv2.rectangle(display, (x1, y1), (x2, y2), (200, 200, 200), 1)

        # Status overlay
        status = ""
        if enrolled:
            status = "✅ Face enrolled! Press S to start virtual camera"
            color = (0, 220, 100)
        elif collecting:
            status = "📸 Collecting frames..."
            color = (0, 200, 255)
        else:
            status = "Press E to enroll your face | Press S to skip enrollment"
            color = (200, 200, 200)

        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Privacy features status
        privacy_status = "🔒 Privacy: Plates, Cards, NSFW, PII will be blurred"
        cv2.putText(display, privacy_status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

        # Virtual camera status
        if VIRTUAL_CAMERA_AVAILABLE:
            cv2.putText(display, "🎥 Virtual Camera Ready", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Instructions
        cv2.putText(display, "E: Enroll | S: Start Camera | Q: Quit", (10, 720-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("censerve — Complete Virtual Camera Setup", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

        elif key == ord('e') and not collecting:
            embeddings_collected = []
            collecting = True
            print("\n📸 [Enrollment] Started collecting frames. Hold still and face the camera.")
            print("🎯 Your face will be whitelisted and stay clear in virtual camera!")

        elif key == ord('s'):
            cap.release()
            cv2.destroyAllWindows()
            break

    # ── Start Complete Virtual Camera Streaming ──
    print("\n" + "="*60)
    print("🚀 Starting Complete Virtual Camera...")
    print("="*60)
    
    if enrolled:
        print("✅ Your enrolled face will NOT be blurred.")
        print("🎯 All other faces WILL be blurred.")
    else:
        print("⚠️  No face enrolled. ALL detected faces will be blurred.")
    
    print("\n🔒 Privacy Protection Active:")
    print("   ❌ License Plates - Blurred")
    print("   ❌ Credit Cards - Blurred")
    print("   ❌ NSFW Content - Blurred")
    print("   ❌ Text PII - Blurred")
    print("   ✅ Your Face - Clear (if enrolled)")
    
    print("\n🎥 Virtual Camera: OBS Virtual Camera")
    print("📱 Compatible: Zoom, Discord, Teams, Meet")
    print("⏹️  Press Q to quit")
    print("="*60)

    # Create virtual video loop
    loop = VirtualVideoLoop(config, source=0, debug=False)

    # Pass the already-enrolled face pipeline into the loop
    loop.face_pipeline = face_pipeline

    # Add object detector for privacy features
    object_detector = PlateCardDetector()
    loop.add_detector(object_detector.detect)

    print("🔍 Object detectors loaded:")
    print("   - License plates (will be blurred)")
    print("   - Credit cards (will be blurred)")
    print("   - NSFW content (will be blurred)")
    print("   - Text PII (will be blurred)")

    try:
        # Start virtual camera stream with all privacy features
        loop.run(use_virtual_camera=True)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure OBS Studio is installed for virtual camera")
    finally:
        loop.stop()
        print("\n🔚 Virtual camera stream stopped")
        print("🎯 Privacy protection disabled")
        print("="*60)

if __name__ == "__main__":
    print("🎭 censerve AI - Complete Privacy Virtual Camera")
    print("="*60)
    
    # Check virtual camera availability
    if not VIRTUAL_CAMERA_AVAILABLE:
        print("⚠️  Warning: Virtual camera not available")
        print("   Install with: pip install pyvirtualcam")
        print("   Or install OBS Studio for virtual camera driver")
        response = input("Continue with local display? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(0)
    
    # Run complete virtual camera
    complete_virtual_camera()
