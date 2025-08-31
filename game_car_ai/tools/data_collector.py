# game_car_ai/tools/data_collector.py
import cv2
import os
import time
import numpy as np
from datetime import datetime
from game_car_ai.src.vision.capture import ScreenCapture

class DataCollector:
    def __init__(self, output_dir="game_car_ai/datasets/car_obstacle/images/train"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Tạo ScreenCapture với kích thước tối đa 800 để dễ label
        self.capture = ScreenCapture(max_size=800)

    def start_collection(self, capture_interval=0.5, max_images=500):
        """Thu thập ảnh từ ScreenCapture"""
        self.capture.start()
        time.sleep(2)  # chờ cho capture ổn định

        count = 0
        print("🎬 Bắt đầu thu thập data... (Nhấn 's' để dừng)")

        try:
            while count < max_images:
                frame = self.capture.get_frame_with_timeout(1.0)
                if frame is None:
                    print("⚠️ Không lấy được frame!")
                    continue

                # Scale về [0,255]
                frame_rgb = frame  
                # Lưu ảnh với timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(self.output_dir, f"car_{timestamp}.jpg")
                cv2.imwrite(filename, frame_rgb)
                count += 1
                print(f"📸 Đã lưu ảnh {count}/{max_images}: {filename}")

                # Preview
                cv2.imshow("Data Collection - Nhấn S để dừng", frame_rgb)

                if cv2.waitKey(1) & 0xFF == ord("s"):
                    break

                time.sleep(capture_interval)

        finally:
            self.capture.stop()
            cv2.destroyAllWindows()
            print(f"✅ Hoàn thành! Đã thu thập {count} ảnh")


def collect_diverse_data():
    collector = DataCollector()
    print("""
    🎯 HƯỚNG DẪN THU THẬP DATA:
    1. Chơi game bình thường 2 phút
    2. Cố tình đâm vào xe khác
    3. Di chuyển các làn đường khác nhau
    4. Tăng tốc và giảm tốc
    5. Thu thập cả day và night scenes (nếu có)
    """)
    input("Nhấn Enter để bắt đầu...")
    collector.start_collection(capture_interval=0.3, max_images=1000)


if __name__ == "__main__":
    collect_diverse_data()
