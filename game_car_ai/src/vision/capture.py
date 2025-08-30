import cv2
import numpy as np
import threading

class ScreenCapture:
    def __init__(self, device_index=10, width=84, height=84):
        self.cap = cv2.VideoCapture(device_index)
        self.width = width
        self.height = height
        if not self.cap.isOpened():
            raise RuntimeError(f"Không thể mở scrcpy virtual camera tại {device_index}")

        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """Luồng đọc frame liên tục"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Tiền xử lý: grayscale + resize + chuẩn hóa
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (self.width, self.height))
                norm = resized.astype(np.float32) / 255.0
                # shape (1, 84, 84)
                processed = np.expand_dims(norm, axis=0)

                with self.lock:
                    self.frame = processed

    def get_frame(self):
        """Lấy frame mới nhất đã tiền xử lý"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Dừng capture"""
        self.running = False
        self.thread.join()
        self.cap.release()
        cv2.destroyAllWindows()
