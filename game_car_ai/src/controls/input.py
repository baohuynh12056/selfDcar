from pynput.keyboard import Controller, Key
import time

keyboard = Controller()

def press_left(duration=0.1):
    keyboard.press(Key.left)
    time.sleep(duration)
    keyboard.release(Key.left)

def press_right(duration=0.1):
    keyboard.press(Key.right)
    time.sleep(duration)
    keyboard.release(Key.right)