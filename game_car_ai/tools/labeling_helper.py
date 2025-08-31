# tools/labeling_helper.py
import os
import cv2
import json
from pathlib import Path

class LabelingHelper:
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        self.current_idx = 0
        self.classes = ['player_car', 'opponent_car']
        
        os.makedirs(label_dir, exist_ok=True)
        
    def start_labeling(self):
        """Giao diện labeling đơn giản"""
        print("🎯 Starting semi-auto labeling...")
        print("Classes: 0=player_car, 1=opponent_car")
        print("Commands: n=next, p=previous, s=skip, q=quit")
        
        while self.current_idx < len(self.images):
            img_name = self.images[self.current_idx]
            img_path = os.path.join(self.image_dir, img_name)
            label_path = os.path.join(self.label_dir, Path(img_name).stem + '.txt')
            
            # Load ảnh
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Cannot read image: {img_name}")
                self.current_idx += 1
                continue
            
            # Hiển thị ảnh
            cv2.imshow('Labeling Helper - Press Q to quit', img)
            
            # Nhập command
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):  # Thoát
                break
            elif key == ord('n'):  # Next
                self.current_idx += 1
            elif key == ord('p'):  # Previous
                self.current_idx = max(0, self.current_idx - 1)
            elif key == ord('s'):  # Skip
                self.current_idx += 1
            else:
                print(f"❌ Unknown command: {chr(key)}")
        
        cv2.destroyAllWindows()

# Chạy helper
if __name__ == "__main__":
    helper = LabelingHelper(
        image_dir='datasets/car_obstacle/images/train/',
        label_dir='datasets/car_obstacle/labels/train/'
    )
    helper.start_labeling()