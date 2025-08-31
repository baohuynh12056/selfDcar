# tools/train_yolo.py
from ultralytics import YOLO
import os

def train_yolo():
    """Train YOLO model"""
    print("🎯 Training YOLO model...")
    
    # Load pretrained model
    model = YOLO('yolov8n.pt')  # YOLOv8 nano
    
    # Train model
    results = model.train(
        data='datasets/car_obstacle/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        project='runs/detect',
        name='car_obstacle_v1',
        optimizer='Adam',
        lr0=0.001,
        augment=True,  # Data augmentation
    )
    
    print("✅ Training completed!")
    return results

if __name__ == "__main__":
    train_yolo()