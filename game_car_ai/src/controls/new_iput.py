# scrcpy_controller.py
import subprocess
import pyautogui
import time
import threading
class ScrcpyController:
    def __init__(self, max_size=640):
        """
        Khởi chạy scrcpy và điều khiển điện thoại bằng PyAutoGUI
        Args:
            max_size: resize window scrcpy
        """
        self.max_size = max_size
        self.proc = None
        self.current_action = None
        self._launch_scrcpy()
        time.sleep(1)  # đợi scrcpy mở xong

    def _launch_scrcpy(self):
        cmd = [
            "scrcpy",
            "--no-playback"
        ]
        self.proc = subprocess.Popen(cmd)

    def press_key(self, key):
        """
        Gửi phím vào cửa sổ scrcpy
        """
        # PyAutoGUI cần window đang active
        pyautogui.press(key)

    def tap(self, x, y):
        """
        Click chuột vào cửa sổ scrcpy, map tọa độ từ scrcpy window
        """
        pyautogui.click(x, y)
    def _tap_key_short(self, key, duration=0.1):
        """Nhấn giữ key trong duration giây rồi nhả"""
        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)

    def swipe(self, x1, y1, x2, y2, duration=0.2):
        pyautogui.moveTo(x1, y1)
        pyautogui.dragTo(x2, y2, duration=duration)
    def _send_key(self, key):
        """Nhả phím cũ và nhấn phím mới"""
        if self.current_action != key:
            self.release()
            pyautogui.keyDown(key)
            self.current_action = key

    def left(self):
        threading.Thread(target=self._tap_key_short, args=("left", 0.1), daemon=True).start()

    def right(self):
        threading.Thread(target=self._tap_key_short, args=("right", 0.1), daemon=True).start()


    def down(self):
        pass

    def noop(self):
        """Nhả tất cả phím đang giữ"""
        pass
    def close(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait()
    def release(self):
        """Nhả phím đang giữ nếu có"""
        if self.current_action:
            pyautogui.keyUp(self.current_action)
            self.current_action = None
