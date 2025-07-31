from vision.capture import capture_screen
from vision.detect import detect_objects
from ai.agent import CarAvoidAgent
from controls.input import press_key

agent = CarAvoidAgent()

while True:
    frame = capture_screen()
    objects = detect_objects(frame)
    action = agent.predict(objects)
    press_key(action)