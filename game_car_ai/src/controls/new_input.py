# src/controls/new_input.py
import subprocess
import pyautogui
import time
import threading

class ScrcpyController:
    def __init__(self, max_size=640):
        """
        Run scrcpy and control the phone using PyAutoGUI
        Args:
            max_size: resize window scrcpy (option)
        """
        self.max_size = max_size
        self.proc = None
        self.current_action = None
        self._launch_scrcpy()
        time.sleep(1)  # Wait until scrcpy finishes lauching

    def _launch_scrcpy(self):
        cmd = [
            "scrcpy",
            "--no-playback"
        ]
        self.proc = subprocess.Popen(cmd)

    def press_key(self, key):
        """
        Send keystrokes to the scrcpy window 
        """
        pyautogui.press(key)

    def tap(self, x, y):
        """
        Click inside the scrcpy window and map the coordinates from it
        """
        pyautogui.click(x, y)

    def _tap_key_short(self, key, duration=0.1):
        """Hold a key for duration seconds, then release"""
        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)

    def swipe(self, x1, y1, x2, y2, duration=0.2):
        pyautogui.moveTo(x1, y1)
        pyautogui.dragTo(x2, y2, duration=duration)

    def _send_key(self, key):
        """Release the previous key before pressing the new one"""
        if self.current_action != key:
            self.release()
            pyautogui.keyDown(key)
            self.current_action = key

    def left(self):
        threading.Thread(target=self._tap_key_short, args=("left", 0.1), daemon=True).start()

    def right(self):
        threading.Thread(target=self._tap_key_short, args=("right", 0.1), daemon=True).start()


    def down(self):
        """
        Do not use the DOWN key — it makes the car stop or run too slowly during training,
        which prevents opponent cars from appearing to avoid
        """
        pass

    def noop(self):
        pass

    def close(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait()

    def release(self):
        """Release the currently held key, if present"""
        if self.current_action:
            pyautogui.keyUp(self.current_action)
            self.current_action = None
