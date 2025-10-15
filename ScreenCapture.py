# pip install pyautogui
import pyautogui

# Capture the entire screen
screenshot = pyautogui.screenshot()

# Save the screenshot to a file
screenshot.save("full_screen_capture.png")

# Capture a specific region (left, top, width, height)
region_screenshot = pyautogui.screenshot(region=(100, 100, 500, 300))
region_screenshot.save("region_capture.png")

