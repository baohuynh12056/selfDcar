# tools/test_realtime.py
from ultralytics import YOLO
import cv2
import numpy as np
from game_car_ai.src.vision.capture import ScreenCapture

def test_realtime_detection():
    # Load model đã train
    model = YOLO('game_car_ai/assets/weights/car_detector.pt')
    capture = ScreenCapture()
    
    print("Starting real-time detection...")
    print("Press 'Q' to quit, 'S' to save screenshot")
    
    try:
        capture.start()
        frame_count = 0
        
        while True:
            frame = capture.get_frame_with_timeout()
            if frame is not None:
                # Convert to BGR
                if len(frame.shape) == 2:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    frame_bgr = frame
                
                # Run detection
                results = model(frame_bgr, conf=0.5, verbose=False)
                
                # Visualize results
                annotated_frame = results[0].plot()
                
                # Hiển thị FPS và thông tin
                cv2.putText(annotated_frame, f"FPS: {30}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, "Press Q to quit", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imshow('Car Detection - Real Time', annotated_frame)
                frame_count += 1
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'detection_screenshot_{frame_count}.jpg', annotated_frame)
                print("📸 Screenshot saved!")
                
    finally:
        capture.stop()
        cv2.destroyAllWindows()
        print("Real-time test completed")

if __name__ == "__main__":
    test_realtime_detection()