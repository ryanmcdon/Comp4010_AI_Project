import win32gui
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import win32api
import keyboard
import datetime

mouse_posy = 0
mouse_posx = 0
mouse_move_amount = 24

def get_active_window_size():
    """
    Get the size of the currently active window.
    Returns a tuple of (width, height) or None if no active window found.
    """
    try:
        # Get the handle of the active window
        active_window = win32gui.GetForegroundWindow()
        
        if active_window == 0:
            print("No active window found")
            return None
        
        # Get the window rectangle (left, top, right, bottom)
        rect = win32gui.GetWindowRect(active_window)
        
        # Calculate width and height
        width = rect[2] - rect[0]  # right - left
        height = rect[3] - rect[1]  # bottom - top
        
        # Get window title for reference
        window_title = win32gui.GetWindowText(active_window)
        
        print(f"Active window: '{window_title}'")
        print(f"Window size: {width} x {height} pixels")
        print(f"Window position: ({rect[0]}, {rect[1]}) to ({rect[2]}, {rect[3]})")
        
        return (width, height)
        
    except Exception as e:
        print(f"Error getting active window size: {e}")
        return None

def get_active_window_info():
    """
    Get detailed information about the active window including size, position, and title.
    """
    try:
        active_window = win32gui.GetForegroundWindow()
        
        if active_window == 0:
            return None
        
        # Get window rectangle
        rect = win32gui.GetWindowRect(active_window)
        
        # Get window title
        title = win32gui.GetWindowText(active_window)
        
        # Get window class name
        class_name = win32gui.GetClassName(active_window)
        
        # Calculate dimensions
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        
        window_info = {
            'title': title,
            'class_name': class_name,
            'width': width,
            'height': height,
            'left': rect[0],
            'top': rect[1],
            'right': rect[2],
            'bottom': rect[3],
            'handle': active_window
        }
        
        return window_info
        
    except Exception as e:
        print(f"Error getting window info: {e}")
        return None

def write_mouse_position_to_file():
    """Write current mouse position to a text file"""
    try:
        global mouse_posy, mouse_posx
        if mouse_posy and mouse_posx:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("mouse_positions.txt", "a") as f:
                f.write(f"[{timestamp}] Mouse Position: ({mouse_posy}, {mouse_posx})\n")
            print(f"Mouse position ({mouse_posy}, {mouse_posx}) written to mouse_positions.txt")
        else:
            print("Unable to get mouse position")
    except Exception as e:
        print(f"Error writing mouse position to file: {e}")

def get_mouse_position():
    """Get current mouse position"""
    try:
        return win32api.GetCursorPos()
    except Exception as e:
        return None

def move_mouse_x_by_amount():
    """Move mouse cursor X position by the global mouse_move_amount"""
    try:
        global mouse_move_amount
        current_pos = get_mouse_position()
        if current_pos:
            new_x = current_pos[0] + mouse_move_amount
            new_y = current_pos[1]
            win32api.SetCursorPos((new_x, new_y))
            print(f"Mouse moved from ({current_pos[0]}, {current_pos[1]}) to ({new_x}, {new_y}) by {mouse_move_amount} pixels")
        else:
            print("Unable to get current mouse position")
    except Exception as e:
        print(f"Error moving mouse: {e}")

def move_mouse_y_by_amount():
    """Move mouse cursor Y position by the global mouse_move_amount"""
    try:
        global mouse_move_amount
        current_pos = get_mouse_position()
        if current_pos:
            new_x = current_pos[0]
            new_y = current_pos[1] + mouse_move_amount
            win32api.SetCursorPos((new_x, new_y))
            print(f"Mouse moved from ({current_pos[0]}, {current_pos[1]}) to ({new_x}, {new_y}) by {mouse_move_amount} pixels")
        else:
            print("Unable to get current mouse position")
    except Exception as e:
        print(f"Error moving mouse: {e}")

def update_window_info():
    """Function to update window information continuously"""
    try:
        # Get window information
        info = get_active_window_info()
        mouse_pos = get_mouse_position()
        
        if info:
            # Clear previous output
            output_text.delete(1.0, tk.END)
            
            # Add timestamp
            timestamp = time.strftime("%H:%M:%S")
            output_text.insert(tk.END, f"[{timestamp}] Active Window Information:\n")
            output_text.insert(tk.END, "=" * 50 + "\n")
            
            # Display window info
            output_text.insert(tk.END, f"Title: {info['title']}\n")
            output_text.insert(tk.END, f"Size: {info['width']} x {info['height']} pixels\n")
            output_text.insert(tk.END, f"Position: ({info['left']}, {info['top']}) to ({info['right']}, {info['bottom']})\n")
            output_text.insert(tk.END, f"Class: {info['class_name']}\n")
            output_text.insert(tk.END, f"Handle: {info['handle']}\n")
            
            # Display mouse position
            if mouse_pos:
                output_text.insert(tk.END, f"Mouse Position: ({mouse_pos[0]}, {mouse_pos[1]})\n")
                
                # Calculate mouse position relative to the active window
                rel_x = mouse_pos[0] - info['left']
                rel_y = mouse_pos[1] - info['top']
                global mouse_posy, mouse_posx
                mouse_posy = rel_y
                mouse_posx = rel_x
                output_text.insert(tk.END, f"Mouse Relative to Window: ({rel_x}, {rel_y})\n")
                
            else:
                output_text.insert(tk.END, "Mouse Position: Unable to get\n")
                
        else:
            output_text.delete(1.0, tk.END)
            timestamp = time.strftime("%H:%M:%S")
            output_text.insert(tk.END, f"[{timestamp}] No active window found\n")
            
            # Still show mouse position even if no active window
            if mouse_pos:
                output_text.insert(tk.END, f"Mouse Position: ({mouse_pos[0]}, {mouse_pos[1]})\n")
            
    except Exception as e:
        output_text.delete(1.0, tk.END)
        timestamp = time.strftime("%H:%M:%S")
        output_text.insert(tk.END, f"[{timestamp}] Error: {e}\n")

def keyboard_listener():
    """Background thread for keyboard listening"""
    while monitoring_active:
        try:
            # Listen for backslash key press
            if keyboard.is_pressed('\\'):
                write_mouse_position_to_file()
                time.sleep(0.1)  # Small delay to prevent multiple rapid triggers
            
            # Listen for up arrow key press to move mouse Y by mouse_move_amount pixels
            if keyboard.is_pressed('up'):
                move_mouse_y_by_amount()
                time.sleep(0.1)  # Small delay to prevent multiple rapid triggers
            
            # Listen for right arrow key press to move mouse X by mouse_move_amount pixels
            if keyboard.is_pressed('right'):
                move_mouse_x_by_amount()
                time.sleep(0.1)  # Small delay to prevent multiple rapid triggers
                
        except Exception as e:
            print(f"Error in keyboard listener: {e}")
        time.sleep(0.01)  # Small delay to prevent high CPU usage

def monitoring_thread():
    """Background thread for continuous monitoring"""
    while monitoring_active:
        update_window_info()
        time.sleep(0.2)  # 5 times per second (200ms interval)

def create_gui():
    """Create the GUI window"""
    global output_text, root, monitoring_active, monitor_thread, mouse_posy, mouse_posx
    
    monitoring_active = True
    
    root = tk.Tk()
    root.title("Window Information Detector - Auto Update (5x/sec)")
    root.geometry("500x400")
    
    # Create info label
    info_label = ttk.Label(root, text="Continuously monitoring active window (5 updates per second)", 
                          font=("Arial", 10))
    info_label.pack(pady=10)
    
    # Create instruction labels
    instruction_label1 = ttk.Label(root, text="Press \\ (backslash) to save current mouse position to file", 
                                  font=("Arial", 9), foreground="blue")
    instruction_label1.pack(pady=2)
    
    instruction_label2 = ttk.Label(root, text=f"Press ↑ (up arrow) to move mouse Y position by {mouse_move_amount} pixels", 
                                  font=("Arial", 9), foreground="green")
    instruction_label2.pack(pady=2)
    
    instruction_label3 = ttk.Label(root, text=f"Press → (right arrow) to move mouse X position by {mouse_move_amount} pixels", 
                                  font=("Arial", 9), foreground="orange")
    instruction_label3.pack(pady=2)
    
    # Create output text area
    output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20)
    output_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    # Add initial message
    output_text.insert(tk.END, "Starting continuous window monitoring...\n")
    output_text.insert(tk.END, "Updates every 200ms (5 times per second)\n")
    output_text.insert(tk.END, "Press \\ (backslash) to save mouse position to mouse_positions.txt\n")
    output_text.insert(tk.END, f"Press ↑ (up arrow) to move mouse Y position by {mouse_move_amount} pixels\n")
    output_text.insert(tk.END, f"Press → (right arrow) to move mouse X position by {mouse_move_amount} pixels\n\n")
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitoring_thread, daemon=True)
    monitor_thread.start()
    
    # Start keyboard listener thread
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    
    # Handle window closing
    def on_closing():
        global monitoring_active
        monitoring_active = False
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    create_gui()
