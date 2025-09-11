import cv2
import numpy as np
import subprocess
from threading import Thread, Lock
import time

class ScreenCapture:
    def __init__(self, device_path="/dev/video10", max_size=800, width=84, height=84):
        self.device_path = device_path
        self.max_size = max_size
        self.width = width
        self.height = height
        self.cap = None
        self.proc = None

        # synchronization variable
        self.lock = Lock()
        self.frame = None
        self.running = False
        self._setup_scrcpy()

    def _setup_scrcpy(self):
        cmd = [
            "scrcpy",
            "--no-playback",
            f"--max-size={self.max_size}",
            f"--v4l2-sink={self.device_path}"
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Wait until scrcpy finishes lauching
        time.sleep(1)

        self.cap = cv2.VideoCapture(self.device_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open scrcpy virtual camera at {self.device_path}")

    def start(self):
        """Start the frame reading thread"""
        self.running = True
        self.thread = Thread(target=self._update_frame, daemon=True)
        self.thread.start()
        return self

    def _update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (640, 640))
                
                with self.lock:
                    self.frame = resized

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def get_frame_with_timeout(self, timeout=1.0):
        """Take frame with timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            frame = self.get_frame()
            if frame is not None:
                return frame
            time.sleep(0.01)
        return None   

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
        cv2.destroyAllWindows()
