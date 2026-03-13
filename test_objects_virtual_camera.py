"""
BlurBerry Objects Detection + Virtual Camera

Runs object detection with virtual camera output for Zoom/Discord integration.
"""
import cv2
import numpy as np
import sys
import time
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared_types import PipelineConfig
from blurberry.video.virtual_video_loop import VirtualVideoLoop
from blurberry.video.object_detector import ObjectDetector

# Virtual camera import
try:
    import pyvirtualcam
    VIRTUAL_CAMERA_AVAILABLE = True
    print("✅ Virtual camera available")
except ImportError:
    VIRTUAL_CAMERA_AVAILABLE = False
    print("⚠️  Virtual camera not available, using local display")

def test_object_detection():
    """Test object detection with virtual camera output"""
    print("\n=== BlurBerry Object Detection + Virtual Camera ===")
    print("This will detect objects and stream to virtual camera for Zoom/Discord")
    print("Press Q to quit\n")

    # Configuration
    config = PipelineConfig(
        blur_faces=False,      # Disable face blur for object detection test
        blur_plates=True,      # Blur license plates
        blur_cards=True,       # Blur credit cards
        blur_nsfw=True,        # Blur NSFW content
        blur_text_pii=True,    # Blur text PII
        detection_cadence=10,  # Run object detection every 10 frames
        blur_strength=51,
        face_similarity_threshold=0.38
    )

    # Create virtual video loop
    loop = VirtualVideoLoop(config, source=0, debug=True)

    # Add object detector
    object_detector = ObjectDetector()
    loop.add_detector(object_detector.detect_objects)

    print("🔍 Object detectors loaded:")
    print("   - License plates")
    print("   - Credit cards")
    print("   - NSFW content")
    print("   - Text PII")

    try:
        # Start virtual camera and object detection
        loop.run(use_virtual_camera=True)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        loop.stop()
        print("Object detection virtual camera stopped")

if __name__ == "__main__":
    print("🎥 BlurBerry AI - Object Detection Virtual Camera")
    print("=" * 60)
    
    # Check virtual camera availability
    if not VIRTUAL_CAMERA_AVAILABLE:
        print("⚠️  Warning: Virtual camera not available")
        print("   Install with: pip install pyvirtualcam")
        print("   Or install OBS Studio for virtual camera driver")
        response = input("Continue with local display? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(0)
    
    # Run object detection test
    test_object_detection()
