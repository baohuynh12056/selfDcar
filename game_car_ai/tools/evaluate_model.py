# tools/evaluate_model.py
from ultralytics import YOLO

def evaluate_model():
    """Đánh giá model sau training"""
    model = YOLO('runs/detect/car_obstacle_v1/weights/best.pt')
    
    # Validation
    metrics = model.val(
        data='datasets/car_obstacle/data.yaml',
        split='val',
        conf=0.5,
        iou=0.5
    )
    
    print(f"📊 Validation metrics:")
    print(f"  mAP50: {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall: {metrics.box.mr:.3f}")

if __name__ == "__main__":
    evaluate_model()