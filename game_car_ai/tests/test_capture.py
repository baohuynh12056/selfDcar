from game_car_ai.src.vision.capture import ScreenCapture
import cv2

def main():
    sc = ScreenCapture(device_path="/dev/video10").start()

    for i in range(10):
        frame = None
        while frame is None:
            frame = sc.get_frame()

        img = (frame[0] * 255).astype("uint8")
        cv2.imwrite(f"frame_{i}.png", img)
        print(f"Frame saved_{i}.png")

    sc.stop()

if __name__ == "__main__":
    main()
