# tools/simple_labeler.py
import cv2
import os
from pathlib import Path

class SimpleLabeler:
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = self.get_image_list()
        self.current_idx = 0
        self.classes = ['player_car', 'opponent_car']
        self.current_class = 0
        self.boxes = []  # list of boxes for current image: (class_id, x1, y1, x2, y2)
        self.drawing = False
        self.start_point = None

        os.makedirs(label_dir, exist_ok=True)
        print(f"📁 Found {len(self.images)} images")

    def get_image_list(self):
        images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            images.extend([f for f in os.listdir(self.image_dir) if f.lower().endswith(ext)])
        return sorted(images)

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for drawing boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = self.start_point
            x2, y2 = x, y
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            self.boxes.append((self.current_class, x1, y1, x2, y2))
            print(f"✅ Added box: class={self.classes[self.current_class]}, ({x1},{y1},{x2},{y2})")

    def save_labels(self, label_path, img_width, img_height):
        """Save boxes in YOLO format"""
        with open(label_path, 'w') as f:
            for cls_id, x1, y1, x2, y2 in self.boxes:
                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                w = (x2 - x1) / img_width
                h = (y2 - y1) / img_height
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        print(f"💾 Saved {len(self.boxes)} boxes to {label_path}")

    def start_labeling(self):
        print("🎯 Simple Labeling Tool")
        print("Controls: 0=player_car 1=opponent_car n=next p=prev s=save u=undo c=clear q=quit")

        while self.current_idx < len(self.images):
            img_name = self.images[self.current_idx]
            img_path = os.path.join(self.image_dir, img_name)
            label_path = os.path.join(self.label_dir, Path(img_name).stem + '.txt')

            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Cannot read: {img_name}")
                self.current_idx += 1
                continue

            self.boxes = []  # reset boxes for this image
            display_img = img.copy()
            h, w = img.shape[:2]

            cv2.namedWindow('Simple Labeler')
            cv2.setMouseCallback('Simple Labeler', self.mouse_callback)

            while True:
                temp_img = display_img.copy()
                # draw all boxes
                for cls_id, x1, y1, x2, y2 in self.boxes:
                    color = (255, 0, 0) if cls_id == 0 else (0, 0, 255)
                    cv2.rectangle(temp_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(temp_img, self.classes[cls_id], (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                cv2.putText(temp_img, f"Class: {self.classes[self.current_class]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(temp_img, f"Image: {self.current_idx + 1}/{len(self.images)}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(temp_img, "0:player_car 1:opponent_car n:next p:prev s:save u:undo c:clear q:quit",
                            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                cv2.imshow('Simple Labeler', temp_img)
                key = cv2.waitKey(20) & 0xFF

                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                elif key == ord('n'):
                    self.current_idx += 1
                    break
                elif key == ord('p'):
                    self.current_idx = max(0, self.current_idx - 1)
                    break
                elif key == ord('0'):
                    self.current_class = 0
                    print("🔵 Selected: player_car")
                elif key == ord('1'):
                    self.current_class = 1
                    print("🔴 Selected: opponent_car")
                elif key == ord('s'):
                    self.save_labels(label_path, w, h)
                elif key == ord('u'):  # undo
                    if self.boxes:
                        removed = self.boxes.pop()
                        print(f"↩️  Removed last box: class={self.classes[removed[0]]}")
                elif key == ord('c'):  # clear all
                    self.boxes = []
                    print("🧹 Cleared all boxes for this image")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    labeler = SimpleLabeler(
        image_dir='datasets/car_obstacle/images/train/',
        label_dir='datasets/car_obstacle/labels/train/'
    )
    labeler.start_labeling()
