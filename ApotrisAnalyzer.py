import win32gui
import win32api
import win32con
import pyautogui
import numpy as np
from PIL import Image
import cv2
import time

class ApotrisAnalyzer:
    def __init__(self, window_name="Apotris PC"):
        self.target_window_name = window_name
        self.window_handle = None
        self.window_rect = None
        self.game_coordinates = None
        
    def list_all_windows(self):
        """
        List all visible windows for debugging
        """
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if window_title:  # Only include windows with titles
                    rect = win32gui.GetWindowRect(hwnd)
                    windows.append((hwnd, window_title, rect))
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        print("Available windows:")
        for i, (hwnd, title, rect) in enumerate(windows):
            print(f"  {i+1}. '{title}' - {rect}")
        
        return windows
    
    def find_window_by_name(self, window_name):
        """
        Find a window by its title/name
        Returns the window handle and rectangle if found
        """
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if window_name.lower() in window_title.lower():
                    windows.append((hwnd, window_title, win32gui.GetWindowRect(hwnd)))
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        if windows:
            # Return the first match
            hwnd, title, rect = windows[0]
            print(f"Found window: '{title}' at {rect}")
            return hwnd, rect
        else:
            print(f"No window found with name containing '{window_name}'")
            print("Available windows:")
            self.list_all_windows()
            return None, None
    
    def capture_window_screenshot(self, window_handle, window_rect):
        """
        Capture a screenshot of the specified window
        """
        try:
            # Bring window to front
            win32gui.SetForegroundWindow(window_handle)
            time.sleep(0.5)  # Give time for window to come to front
            
            # Capture the window region
            left, top, right, bottom = window_rect
            width = right - left
            height = bottom - top
            
            # Take screenshot of the specific region
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            
            # Save for debugging
            screenshot.save("apotris_capture.png")
            print(f"Screenshot saved as 'apotris_capture.png'")
            
            return screenshot
            
        except Exception as e:
            print(f"Error capturing window screenshot: {e}")
            return None
    
    def analyze_for_game_area(self, screenshot):
        """
        Analyze the screenshot to find game area surrounded by black pixels
        Improved to handle window borders and UI elements
        """
        try:
            # Convert PIL image to numpy array
            img_array = np.array(screenshot)
            height, width = img_array.shape[:2]
            
            # Convert to grayscale for easier analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Method 1: Look for rectangular game area with black borders
            game_coords = self._find_game_area_by_borders(gray, width, height)
            
            if game_coords:
                print(f"Game area found using border detection:")
                print(f"  Top-left: {game_coords['top_left']}")
                print(f"  Bottom-right: {game_coords['bottom_right']}")
                print(f"  Center: {game_coords['center']}")
                print(f"  Size: {game_coords['width']}x{game_coords['height']} pixels")
                print(f"  Area: {game_coords['area']} pixels")
                return game_coords

            print("No game area found with any detection method")
            return None
                
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None
    
    def _find_game_area_by_borders(self, gray_image, width, height):
        """
        Find game area by detecting black borders around a rectangular region
        """
        try:
            # Define black threshold (adjust based on your game)
            black_threshold = 50
            
            # Skip window borders - start analysis from inner area
            border_margin = 50  # Skip first 50 pixels from edges
            start_x = border_margin
            start_y = border_margin
            end_x = width - border_margin
            end_y = height - border_margin
            
            # Look for horizontal black lines (top and bottom borders)
            top_border_y = None
            bottom_border_y = None
            
            # Scan from top to find first significant black line
            for y in range(start_y, end_y - 100):  # Leave room for bottom border
                row = gray_image[y, start_x:end_x]
                black_pixels = np.sum(row < black_threshold)
                total_pixels = len(row)
                
                # If more than 95% of pixels in this row are black, it's likely a border
                if black_pixels / total_pixels > 0.95:
                    top_border_y = y
                    break
            
            # Scan from bottom to find first significant black line
            for y in range(end_y - 1, start_y + 100, -1):  # Leave room for top border
                row = gray_image[y, start_x:end_x]
                black_pixels = np.sum(row < black_threshold)
                total_pixels = len(row)
                
                if black_pixels / total_pixels > 0.95:
                    bottom_border_y = y
                    break
            
            # Look for vertical black lines (left and right borders)
            left_border_x = None
            right_border_x = None
            
            if top_border_y and bottom_border_y:
                # Scan from left to find first significant black line
                for x in range(start_x, end_x - 100):
                    col = gray_image[top_border_y:bottom_border_y, x]
                    black_pixels = np.sum(col < black_threshold)
                    total_pixels = len(col)
                    
                    if black_pixels / total_pixels > 0.95:
                        left_border_x = x
                        break
                
                # Scan from right to find first significant black line
                for x in range(end_x - 1, start_x + 100, -1):
                    col = gray_image[top_border_y:bottom_border_y, x]
                    black_pixels = np.sum(col < black_threshold)
                    total_pixels = len(col)
                    
                    if black_pixels / total_pixels > 0.95:
                        right_border_x = x
                        break
            
            # If we found all four borders, calculate game area
            if all([top_border_y, bottom_border_y, left_border_x, right_border_x]):
                # Add small margin inside the borders to get actual game area
                margin = 5
                game_x = left_border_x + margin
                game_y = top_border_y + margin
                game_w = right_border_x - left_border_x - 2 * margin
                game_h = bottom_border_y - top_border_y - 2 * margin
                
                # Validate that the detected area is reasonable
                if game_w > 100 and game_h > 100 and game_w < width * 0.9 and game_h < height * 0.9:
                    center_x = game_x + game_w // 2
                    center_y = game_y + game_h // 2
                    
                    return {
                        'top_left': (game_x, game_y),
                        'bottom_right': (game_x + game_w, game_y + game_h),
                        'center': (center_x, center_y),
                        'width': game_w,
                        'height': game_h,
                        'area': game_w * game_h
                    }
            
            return None
            
        except Exception as e:
            print(f"Error in border detection: {e}")
            return None

    
    def visualize_game_area(self, screenshot, game_coords):
        """
        Create a visualization showing the detected game area
        """
        try:
            # Convert PIL image to numpy array
            img_array = np.array(screenshot)
            
            # Draw rectangle around game area
            cv2.rectangle(img_array, 
                         game_coords['top_left'], 
                         game_coords['bottom_right'], 
                         (0, 255, 0),  # Green color
                         3)  # Line thickness
            
            # Draw center point
            cv2.circle(img_array, 
                      game_coords['center'], 
                      5, 
                      (255, 0, 0),  # Red color
                      -1)  # Filled circle
            
            # Add text labels
            cv2.putText(img_array, f"Game Area: {game_coords['width']}x{game_coords['height']}", 
                       (game_coords['top_left'][0], game_coords['top_left'][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert back to PIL image
            result_image = Image.fromarray(img_array)
            result_image.save("apotris_analysis.png")
            print("Analysis visualization saved as 'apotris_analysis.png'")
            
            return result_image
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None
    
    def create_debug_visualization(self, screenshot):
        """
        Create debug visualization showing border detection process
        """
        try:
            img_array = np.array(screenshot)
            height, width = img_array.shape[:2]
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Create debug image
            debug_img = img_array.copy()
            
            # Draw border margin lines
            border_margin = 50
            cv2.line(debug_img, (border_margin, 0), (border_margin, height), (255, 0, 255), 2)  # Left margin
            cv2.line(debug_img, (width - border_margin, 0), (width - border_margin, height), (255, 0, 255), 2)  # Right margin
            cv2.line(debug_img, (0, border_margin), (width, border_margin), (255, 0, 255), 2)  # Top margin
            cv2.line(debug_img, (0, height - border_margin), (width, height - border_margin), (255, 0, 255), 2)  # Bottom margin
            
            # Add text
            cv2.putText(debug_img, "Purple lines: Border margins (50px)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(debug_img, f"Image size: {width}x{height}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save debug image
            debug_image = Image.fromarray(debug_img)
            debug_image.save("apotris_debug.png")
            print("Debug visualization saved as 'apotris_debug.png'")
            
            return debug_image
            
        except Exception as e:
            print(f"Error creating debug visualization: {e}")
            return None
    
    def run_analysis(self):
        """
        Main method to run the complete analysis
        """
        print(f"Looking for window: '{self.target_window_name}'")
        
        # Step 1: Find the window
        self.window_handle, self.window_rect = self.find_window_by_name(self.target_window_name)
        
        if not self.window_handle:
            print(f"Could not find window '{self.target_window_name}'")
            return False
        
        # Step 2: Capture screenshot
        print("Capturing window screenshot...")
        screenshot = self.capture_window_screenshot(self.window_handle, self.window_rect)
        
        if not screenshot:
            print("Failed to capture screenshot")
            return False
        
        # Step 2.5: Create debug visualization
        print("Creating debug visualization...")
        self.create_debug_visualization(screenshot)
        
        # Step 3: Analyze for game area
        print("Analyzing image for game area...")
        self.game_coordinates = self.analyze_for_game_area(screenshot)
        
        if not self.game_coordinates:
            print("No game area detected")
            return False
        
        # Step 4: Create visualization
        print("Creating visualization...")
        self.visualize_game_area(screenshot, self.game_coordinates)
        
        # Step 5: Convert to screen coordinates
        screen_coords = self.convert_to_screen_coordinates(self.game_coordinates)
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        print(f"Window: '{self.target_window_name}'")
        print(f"Window position: {self.window_rect}")
        print(f"Game area (relative to window):")
        print(f"  Top-left: {self.game_coordinates['top_left']}")
        print(f"  Bottom-right: {self.game_coordinates['bottom_right']}")
        print(f"  Center: {self.game_coordinates['center']}")
        print(f"Game area (screen coordinates):")
        print(f"  Top-left: {screen_coords['top_left']}")
        print(f"  Bottom-right: {screen_coords['bottom_right']}")
        print(f"  Center: {screen_coords['center']}")
        
        return True
    
    def convert_to_screen_coordinates(self, game_coords):
        """
        Convert window-relative coordinates to screen coordinates
        """
        window_left, window_top = self.window_rect[0], self.window_rect[1]
        
        screen_coords = {
            'top_left': (game_coords['top_left'][0] + window_left, 
                        game_coords['top_left'][1] + window_top),
            'bottom_right': (game_coords['bottom_right'][0] + window_left, 
                           game_coords['bottom_right'][1] + window_top),
            'center': (game_coords['center'][0] + window_left, 
                      game_coords['center'][1] + window_top),
            'width': game_coords['width'],
            'height': game_coords['height'],
            'area': game_coords['area']
        }
        
        return screen_coords

if __name__ == "__main__":
    # Example usage: analyzer = ApotrisAnalyzer(); analyzer.run_analysis()
    analyzer = ApotrisAnalyzer()
    analyzer.run_analysis()
