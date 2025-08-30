import torch
import cv2

class CarDectector:
    def __init__(self, model_name="yoloy8", target_classes=None):
        print("[INFO] Đang load model YOLO...")
        self.model = torch.hub.load('ultralytics/yolov8', model_name, pretrained=True)
        self.model.conf = 0.4  # độ tin cậy tối thiểu
        self.target_classes = target_classes

    def detect(self, frame):
        results = self.model(frame)

        # Lọc kết quả nếu cần
        if self.target_classes:
            results = results.pandas().xyxy[0]
            results = results[results['name'].isin(self.target_classes)]
        else:
            results = results.pandas().xyxy[0]

        # Vẽ bounding box
        annotated_frame = frame.copy()
        for _, row in results.iterrows():
            x1, y1, x2, y2, conf, cls_name = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['name']
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return annotated_frame, results