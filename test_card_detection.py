"""
Test Credit Card Detection (Shape-Based)
"""
import cv2
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from censerve.video.object_detector import PlateCardDetector

def test_card_detection():
    """Test credit card detection with shape-based fallback"""
    print("🔍 Testing Credit Card Detection...")
    
    # Initialize detector
    detector = PlateCardDetector()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("📸 Show a credit card to the camera")
    print("🎯 Shape-based detection should find card-like rectangles")
    print("❌ Press Q to quit")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect objects (every 10 frames to save processing)
        if frame_count % 10 == 0:
            events = detector.detect(frame, frame_count)
            
            # Draw detection results
            for event in events:
                if event.type == "card":
                    x1, y1, x2, y2 = event.bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "CARD DETECTED", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    print(f"✅ Card detected at frame {frame_count}")
        
        # Show frame
        cv2.imshow("Credit Card Detection Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("🔚 Test complete")

if __name__ == "__main__":
    test_card_detection()
