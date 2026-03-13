"""
Debug Credit Card Detection
"""
import cv2
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared_types import PipelineConfig
from censerve.video.object_detector import PlateCardDetector

def debug_card_detection():
    """Debug credit card detection"""
    print("🔍 Debugging Credit Card Detection...")
    
    # Initialize detector
    detector = PlateCardDetector()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("📸 Show a credit card to camera")
    print("🎯 This will show ALL detections (plates, cards)")
    print("❌ Press Q to quit")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect objects EVERY frame for debugging
        events = detector.detect(frame, frame_count)
        
        print(f"Frame {frame_count}: Found {len(events)} objects")
        
        # Draw ALL detection results
        for i, event in enumerate(events):
            x1, y1, x2, y2 = event.bbox
            
            if event.type == "card":
                color = (0, 255, 0)  # Green for cards
                label = "CARD DETECTED"
                print(f"  ✅ Card {i+1}: {event.bbox}")
            elif event.type == "plate":
                color = (255, 0, 0)  # Blue for plates
                label = "PLATE DETECTED"
                print(f"  ✅ Plate {i+1}: {event.bbox}")
            else:
                color = (0, 0, 255)  # Red for others
                label = f"{event.type.upper()}"
                print(f"  ❓ Other {i+1}: {event.type} - {event.bbox}")
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show frame
        cv2.imshow("Debug - Card/Plate Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("🔚 Debug complete")

if __name__ == "__main__":
    debug_card_detection()
