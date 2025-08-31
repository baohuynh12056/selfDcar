# tools/validate_dataset.py
import os
import cv2
import numpy as np
from pathlib import Path

def validate_dataset(image_dir, label_dir, class_names):
    """Kiểm tra dataset đã label"""
    print("🔍 Validating dataset...")
    
    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, Path(img_name).stem + '.txt')
        
        # Kiểm tra ảnh tồn tại
        if not os.path.exists(img_path):
            print(f"❌ Missing image: {img_name}")
            continue
            
        # Kiểm tra label tồn tại
        if not os.path.exists(label_path):
            print(f"⚠️  Missing label: {img_name}")
            continue
            
        # Kiểm tra label format
        with open(label_path, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"❌ Invalid label in {img_name}, line {i+1}")
                
                class_id = int(parts[0])
                if class_id >= len(class_names):
                    print(f"❌ Invalid class ID in {img_name}: {class_id}")
    
    print("✅ Dataset validation completed!")

# Chạy validation
validate_dataset(
    image_dir='datasets/car_obstacle/images/train/',
    label_dir='datasets/car_obstacle/labels/train/',
    class_names=['player_car', 'opponent_car']
)