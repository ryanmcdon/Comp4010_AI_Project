import win32gui
import win32api
import win32con
import pyautogui
import numpy as np
from PIL import Image
import cv2
import time
print_flag = False #can affect more latter but at implementation only effects colour printout
global_height = 0 #global height of the board

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
        
        if print_flag:
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
            if print_flag:
                print(f"Found window: '{title}' at {rect}")
            return hwnd, rect
        else:
            if print_flag:
                print(f"No window found with name containing '{window_name}'")
                print("Available windows:")
                self.list_all_windows()
                return None, None
            else:
                self.list_all_windows()
                return None, None
    
    def capture_window_screenshot(self, window_handle, window_rect):
        """
        Capture a screenshot of the specified window
        """
        try:
            # Bring window to front
            win32gui.SetForegroundWindow(window_handle)
            #time.sleep(0.5)  # Give time for window to come to front
            
            # Capture the window region
            left, top, right, bottom = window_rect
            width = right - left
            height = bottom - top
            
            # Take screenshot of the specific region
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            
            # Save for debugging
            #screenshot.save("apotris_capture.png")
            #print(f"Screenshot saved as 'apotris_capture.png'")
            
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
                # Save coordinates to persistent variable
                self.game_coordinates = game_coords
                
                if print_flag:
                    print(f"Game area found using border detection:")
                    print(f"  Top-left: {game_coords['top_left']}")
                    print(f"  Bottom-right: {game_coords['bottom_right']}")
                    print(f"  Center: {game_coords['center']}")
                    print(f"  Size: {game_coords['width']}x{game_coords['height']} pixels")
                    print(f"  Area: {game_coords['area']} pixels")
                return game_coords
            else:
                if print_flag:
                    print("No game area found with any detection method")
                return None
                
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None
    
    
    """
    Find game area by detecting where non-black pixels start (game content)
    """
    def _find_game_area_by_borders(self, gray_image, width, height):
        # Define black threshold (adjust based on your game)
        black_threshold = 50 # the value of a black pixel
        threshold = 0.2
        
        # Skip window borders - start analysis from inner area
        border_margin = 50  # Skip first 50 pixels from edges
        start_x = border_margin
        start_y = border_margin
        end_x = width - border_margin
        end_y = height - border_margin
        
        # Look for where game content starts (non-black pixels)
        game_start_y = None
        game_end_y = None
        
        # Scan from top to find first significant non-black line (game content starts)
        for y in range(start_y, end_y - border_margin*2):
            row = gray_image[y, start_x:end_x]
            non_black_pixels = np.sum(row >= black_threshold)
            total_pixels = len(row)
            
            # If more than 20% of pixels in this row are non-black, game content likely starts here
            if non_black_pixels / total_pixels > threshold:
                game_start_y = y
                break
        
        # Scan from bottom to find last significant non-black line (game content ends)
        for y in range(end_y - 1, start_y + border_margin*2, -1):
            row = gray_image[y, start_x:end_x]
            non_black_pixels = np.sum(row >= black_threshold)
            total_pixels = len(row)
            
            if non_black_pixels / total_pixels > threshold:
                game_end_y = y
                break
        
        # Look for where game content starts horizontally (left and right edges)
        game_start_x = None
        game_end_x = None
        
        # Scan from left to find first significant non-black line
        for x in range(start_x, end_x - border_margin*2):
            col = gray_image[game_start_y:game_end_y, x]
            non_black_pixels = np.sum(col >= black_threshold)
            total_pixels = len(col)
            
            if non_black_pixels / total_pixels > threshold:
                game_start_x = x
                break
        
        # Scan from right to find last significant non-black line
        for x in range(end_x - 1, start_x + border_margin*2, -1):
            col = gray_image[game_start_y:game_end_y, x]
            non_black_pixels = np.sum(col >= black_threshold)
            total_pixels = len(col)
            
            if non_black_pixels / total_pixels > threshold:
                game_end_x = x
                break
        

        game_x = game_start_x + 1
        game_y = game_start_y + 1
        game_w = game_end_x - game_start_x - 2 * 1
        game_h = game_end_y - game_start_y - 2 * 1
        

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



    """
    Create a visualization showing the detected game area
    """
    def visualize_game_area(self, screenshot, game_coords):

        try:
            # Convert PIL image to numpy array
            img_array = np.array(screenshot)
            
            # Draw rectangle around game area
            cv2.rectangle(img_array, 
                         game_coords['top_left'], 
                         game_coords['bottom_right'], 
                         (0, 255, 0),  # Green color
                         3)  # Line thickness
            
            # Find the top left corner of tetris board
            x , y = game_coords['top_left']
            offset = 4
            x+=273 # center on block
            y+=224+offset
            separation = 10 
            # Draw point
            colors = [] # array of colors 
            for i in range(20):
                for j in range(10): 
                    # Get the color at the current (x, y) pixel
                    colors.append(img_array[y, x].tolist())
                    cv2.circle(img_array, 
                            (x, y), 
                            1, 
                            (255, 0, 0),  # Red color
                            -1)  # Filled circle
                    x+=separation
                x = game_coords['top_left'][0]+273
                y+=separation
        
            file = open("pixel_colors.txt", "w")  
            string = ""
            for i in range(len(colors)):
                if i%10 == 0 and i != 0:
                    if print_flag:
                        print(string)
                    file.write(string + "\n")
                    string = ""
                string += str(colors[i]) + "\t"
            if print_flag:
                print(string)
            file.write(string)
            file.close()
            

            
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
    
    
    #finds if colour is blackground or not (third pixel value of is 42 if backgroud, added some buffer for error)
    def is_backgroud(self, colour):
        if colour[2] >= 41 and colour[2]<=55:
            return True
        return False
    
    #takes in screenshot and game coords returns a board of active pieces
    def get_board(self, screenshot, game_coords):
        img_array = np.array(screenshot)
        x , y = game_coords['top_left']
        x+=273# center on block
        y+=224
        separation = 10 
        board = [] 
        for j in range(10):
            for i in range(20):
                square = self.is_backgroud(img_array[y, x])
                board.append(square)
                y+=separation
            x+=separation
            y=game_coords['top_left'][1]+224
        return board
    
    def is_not_white(self, colour):
        if colour[0] >= 220 and colour[1] >= 220 and colour[2] >= 220:
            return False
        return True
    
    def get_board_white(self, screenshot, game_coords):
        img_array = np.array(screenshot)
        x , y = game_coords['top_left']
        offset = 4
        x+=273# check for where white boarder will be
        y+=224+offset
        separation = 10 
        board = [] 
        for j in range(10):
            for i in range(20):
                square = self.is_not_white(img_array[y, x].tolist())
                board.append(square)
                y+=separation
            x+=separation
            y=game_coords['top_left'][1]+224+offset
        return board
    
    def print_board(self, board):
        for i in range(len(board)):
            if i%20 == 0 and i != 0:
                print()
            print(board[i], end="\t")
        print()
        
    def countour_detection(self, board):
        contour = []
        height = 0
        column = False
        for i in range(10):
            for j in range(20):
                if i == 0 and board[i*20+j] == False and column == False:
                    contour.append(j)
                    column = True
                    global global_height
                    global_height = j
                    height = j
                if board[i*20+j] == False and column == False:
                    contour.append(max(min(height-j, 4), -4))
                    height = j
                    column = True
            if column == True:
                column = False
            else:
                contour.append(max(min(height-i, 4), -4))
                height = 0
        # print(contour)
        return contour
        
    #takes in a board and returns the contour
    #finds the contour of the board, returns active piece and board
    #change to take in a board and return the contour
    # def countour_detection(self, screenshot, game_coords):
            # img_array = np.array(screenshot)
            # x , y = game_coords['top_left']
            # x+=273# center on block
            # y+=224
            # separation = 10 
            # board = [] 
            # contour = []
            # height = 0
            # column = False
            # for j in range(10):
            #     for i in range(20): 
            #         # Get the color at the current (x, y) pixel
            #         square = self.is_backgroud(img_array[y, x])
            #         board.append(square)
            #         if square == False and j == 0 and column == False:
            #             height = i
            #             column = True
            #         elif square == False and j > 0 and column == False:
            #             contour.append( max(min(height-i, 4), -4))
            #             height = i
            #             column = True
            #         y+=separation
            #     if column == True:
            #         column = False
            #     else:
            #         contour.append(max(min(height-i, 4), -4))
            #         height = 0

            #     x+=separation
            #     y=game_coords['top_left'][1]+224

            # print(contour)
            # #file = open("board.txt", "w")  
            # string = ""
            # for i in range(len(board)):
            #     if i%20 == 0 and i != 0:
            #         print(string)
            #         #file.write(string + "\n")
            #         string = ""
            #     if board[i] == True:
            #         string += " - " + "\t"
            #     else:
            #         string += "Block" + "\t"
            # print(string)
            # #file.write(string)
            # #file.close()
            
            # self.create_binary_board(board)
            # return board

    def create_binary_board(self, board):
        grid_rows = [board[i*10:(i+1)*10] for i in range(20)]

        binary_grid = [[1 if cell else 0 for cell in row] for row in grid_rows]

        print("Current board (1 = block, 0 = empty):")
        for row in binary_grid:
            print("".join(str(x) for x in row))
        return binary_grid


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
    
    #Main method to run the complete analysis without visualization
    #used to simplify process for repeated calls
    
    def get_board_state(self):
        print_flag = False
        self.window_handle, self.window_rect = self.find_window_by_name(self.target_window_name)
        screenshot = self.capture_window_screenshot(self.window_handle, self.window_rect)
        self.game_coordinates = self.analyze_for_game_area(screenshot)
        board = self.get_board(screenshot, self.game_coordinates)
        contour = self.countour_detection(board)
        return contour
    
    def run_analysis_no_visualization(self):
        print_flag = False
        
        print(f"Looking for window: '{self.target_window_name}'")
        
        # Step 1: Find the window
        self.window_handle, self.window_rect = self.find_window_by_name(self.target_window_name)
        
        if not self.window_handle:
            print(f"Could not find window '{self.target_window_name}'")
            return False
        
        screenshot = self.capture_window_screenshot(self.window_handle, self.window_rect)
        
        if not screenshot:
            print("Failed to capture screenshot")
            return False
        
        # Step 2: Analyze for game area
        self.game_coordinates = self.analyze_for_game_area(screenshot)
        
        if not self.game_coordinates:
            print("No game area detected")
            return False
        
        #step 3: Analyse for pieces and contour
        white_board = self.get_board_white(screenshot, self.game_coordinates)
        contour = self.countour_detection(white_board)
        board = self.get_board(screenshot, self.game_coordinates)
        
        return {
            'game_coordinates': self.game_coordinates,
            'white_board': white_board,
            'contour': contour,
            'board': board
        }
        
    
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
        #print("Creating debug visualization...")
        #self.create_debug_visualization(screenshot)
        
        # Step 3: Analyze for game area
        print("Analyzing image for game area...")
        self.game_coordinates = self.analyze_for_game_area(screenshot)
        
        if not self.game_coordinates:
            print("No game area detected")
            return False
        
        # Step 4: Create visualization
        print("Creating visualization...")
        # board = self.get_board(screenshot, self.game_coordinates)
        # self.print_board(board)
        self.visualize_game_area(screenshot, self.game_coordinates)
        # self.countour_detection(board)
        white_board = self.get_board_white(screenshot, self.game_coordinates)
        self.print_board(white_board)
        contour = self.countour_detection(white_board)
        
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
        
        return {
            'game_coordinates': self.game_coordinates,
            'white_board': white_board,
            'contour': contour
        }
    
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

    def get_global_height():
        return global_height

if __name__ == "__main__":
    # Example usage: analyzer = ApotrisAnalyzer(); analyzer.run_analysis()
    analyzer = ApotrisAnalyzer()
    analyzer.run_analysis()
