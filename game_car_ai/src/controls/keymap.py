# src/controls/keymap.py
from .new_iput import ScrcpyController

class GameKeyMap:
    def __init__(self, controller):
        self.controller = controller
        self.current_action = None
        self.ACTION_MAP = {
            0: self.left,
            1: self.right,
            2: self.down,
            3: self.noop,
        }

    def left(self):
        self.controller.left()
     

    def right(self):
        self.controller.right()

    def down(self):
        self.controller.down()

    def noop(self):
        self.controller.noop()

    def execute(self, action_id):
        self.ACTION_MAP.get(action_id, self.noop)()
# For RL agent (4 actions)
DRIVING_ACTIONS = list(range(4))
