import time
import win32api
import win32con
import win32gui

VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_UP = 0x26
VK_SPACE = 0x20
VK_Z = 0x5A
VK_A = 0x41
VK_C = 0x43
VK_ESC = 0x1B

GAME_WINDOW_TITLE = "Apotris PC"  

def focus_apotris():
    hwnd = win32gui.FindWindow(None, GAME_WINDOW_TITLE)
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(0.1)


def press_key(vk_code, delay=0.05):
    win32api.keybd_event(vk_code, 0, 0, 0)
    time.sleep(delay)
    win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)

class BotInput:
    @staticmethod
    def move_left():
         
        print("[BotInput] Pressing LEFT")
        press_key(VK_LEFT)

    @staticmethod
    def move_right():
         
        print("[BotInput] Pressing RIGHT")
        press_key(VK_RIGHT)

    @staticmethod
    def rotate_left():
         
        print("[BotInput] Pressing Z")
        press_key(VK_Z)

    @staticmethod
    def rotate_right():
         
        print("[BotInput] Pressing UP")
        press_key(VK_UP)

    @staticmethod
    def rotate_180():
         
        print("[BotInput] Pressing A")
        press_key(VK_A)

    @staticmethod
    def hard_drop():
         
        print("[BotInput] Pressing SPACE")
        press_key(VK_SPACE)

    @staticmethod
    def hold():
         
        print("[BotInput] Pressing C")
        press_key(VK_C)
        
    @staticmethod
    def pause():
        print("[BotInput] Pressing ESC")
        press_key(VK_ESC)
