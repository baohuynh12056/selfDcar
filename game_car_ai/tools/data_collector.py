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
        self.capture = ScreenCapture()

    def start_collection(self, capture_interval=0.5, max_images=500):
        """Collect images from ScreenCapture"""
        self.capture.start()
        time.sleep(2)  

        count = 0
        print("Starting data collection... (Press 's' to stop)")

        try:
            while count < max_images:
                frame = self.capture.get_frame_with_timeout(1.0)
                if frame is None:
                    print("Failed to get frame!")
                    continue

                frame_rgb = frame  
                # Save image with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(self.output_dir, f"car_{timestamp}.jpg")
                cv2.imwrite(filename, frame_rgb)
                count += 1
                print(f"Saved image {count}/{max_images}: {filename}")

                # Preview
                cv2.imshow("Data Collection - Press S to stop", frame_rgb)

                if cv2.waitKey(1) & 0xFF == ord("s"):
                    break

                time.sleep(capture_interval)

        finally:
            self.capture.stop()
            cv2.destroyAllWindows()
            print(f"Completed! Collected {count} images")


def collect_diverse_data():
    collector = DataCollector()
    input("Press Enter to start ...")
    collector.start_collection(capture_interval=0.3, max_images=1000)


if __name__ == "__main__":
    collect_diverse_data()
