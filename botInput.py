import pyautogui
import time


INPUT_DELAY = 0.05

class BotInput:
    def move_left(times=1):
        for _ in range(times):
            pyautogui.press('left')
            time.sleep(INPUT_DELAY)


    def move_right(times=1):
        for _ in range(times):
            pyautogui.press('right')
            time.sleep(INPUT_DELAY)


    def rotate_left(times=1):
        for _ in range(times):
            pyautogui.press('z')
            time.sleep(INPUT_DELAY)


    def rotate_right(times=1):
        for _ in range(times):
            pyautogui.press('up')
            time.sleep(INPUT_DELAY)


    def rotate_180(times=1):
        for _ in range(times):
            pyautogui.press('a')
            time.sleep(INPUT_DELAY)


    def hard_drop():
        pyautogui.press('space')
        time.sleep(INPUT_DELAY)

    def hold():
        pyautogui.press('c')
        time.sleep(INPUT_DELAY)
