from ultralytics import YOLO
import cv2
import numpy as np

class CarDetector:
    def __init__(self, model_path='game_car_ai/assets/weights/car_detector.pt'):
        try:
            self.model = YOLO(model_path)
            self.class_names = {0: 'player_car', 1: 'opponent_car', 2:'menu_game'}
            print(f"CarDetector loaded: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    def detect(self, frame):
        if frame is None:
            return []
        
        try:
            # Convert to BGR nếu là grayscale
            if len(frame.shape) == 2:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 1:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = frame
            
            # Run detection
            results = self.model(frame_bgr, conf=0.5, verbose=False)
            
            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': class_id,
                        'class_name': self.class_names.get(class_id, 'unknown')
                    })
            
            return detections
            
        except Exception as e:
            print(f" Detection error: {e}")
            return []
        
    def visualize(self, frame, detections):
        """
        Vẽ bounding boxes lên frame
        Args:
            frame: Input frame
            detections: List detections từ detect()
        Returns:
            Frame với bounding boxes
        """
        if frame is None:
            return None
        
        # Convert to BGR for visualization
        if len(frame.shape) == 2:
            vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            vis_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            
            # Chọn màu theo class
            if det['class_name'] == 'player_car':
                color = (0, 255, 0)  # Xanh lá
            else:
                color = (0, 0, 255)  # Đỏ
            
            # Vẽ bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ label
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(vis_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame


