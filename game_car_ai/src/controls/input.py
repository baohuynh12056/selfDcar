from pynput.keyboard import Controller
import time

class AIController:
    def __init__(self):
        self.keyboard = Controller()
    
    def press_key(self, key, duration=0.1):
        """Nhấn phím ảo (trái/phải)"""
        self.keyboard.press(key)
        time.sleep(duration)
        self.keyboard.release(key)