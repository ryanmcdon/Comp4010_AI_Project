import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import pyautogui
from ApotrisAnalyzer import ApotrisAnalyzer
from botInput import BotInput, focus_apotris


class env(gym.Env):
    def __init__(self):
        super().__init__()
        self.analyzer = ApotrisAnalyzer()
        self.cached_region = None  
        self.x_offset = 273  
        self.y_offset = 224
        self.cell_spacing = 10

        self.action_space = spaces.Discrete(7)
        self.state_space = spaces.Box(
            low=-4, 
            high=4, 
            shape=(10,),  # 10 columns x 20 rows = 200 flattened
            dtype=np.int8
        )

    def reset(self, seed=None, options=None):
        focus_apotris()

        info = self.analyzer.run_analysis_no_visualization()
        if not info:
            raise RuntimeError(
                "Failed to detect game window or screenshot. Make sure the game is open and visible."
            )

        self.cached_region = info["game_coordinates"]
        self.region_tuple = (
            self.cached_region["top_left"][0],
            self.cached_region["top_left"][1],
            self.cached_region["width"],
            self.cached_region["height"],
        )


        self.top_left_x = self.cached_region["top_left"][0] + self.x_offset
        self.top_left_y = self.cached_region["top_left"][1] + self.y_offset

        contour = np.array(info["contour"], dtype=np.int8)
        return contour, {}

    def step(self, action):
        if action == 0:
            BotInput.move_left()
        elif action == 1:
            BotInput.move_right()
        elif action == 2:
            BotInput.rotate_left()
        elif action == 3:
            BotInput.rotate_right()
        elif action == 4:
            BotInput.rotate_180()
        elif action == 5:
            BotInput.hard_drop()
        elif action == 6:
            BotInput.hold()
        time.sleep(0.02)

        board_state = self.analyzer.get_board_state()


        reward = 0.0
        done = False
        return np.array(board_state, dtype=np.int8), reward, done, False, {}
