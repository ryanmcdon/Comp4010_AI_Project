#!/usr/bin/env python3
"""
Main Program for Comp4010 AI Project
====================================

This is the main entry point for the screen analysis and capture tools.
It provides a unified interface to access all the different tools and functionalities.

Course: Comp4010 AI Project
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import subprocess
import webbrowser
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ScreenCapture import *
    from ScreenInfoDisplay import *
    from ScreenOverlay import *
    from ApotrisAnalyzer import ApotrisAnalyzer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required dependencies are installed:")
    print("pip install -r requirements.txt")

class MainApplication:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Comp4010 AI Project - Screen Analysis Tools")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Set window icon if available
        try:
            self.root.iconbitmap("assets/garbage.png")
        except:
            pass
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main user interface"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Comp4010 AI Project", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        subtitle_label = ttk.Label(main_frame, text="Screen Analysis & Capture Tools", 
                                  font=("Arial", 12))
        subtitle_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # Tool selection frame
        tools_frame = ttk.LabelFrame(main_frame, text="Available Tools", padding="10")
        tools_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        tools_frame.columnconfigure(1, weight=1)
        
        # Tool buttons
        self.create_tool_button(tools_frame, 0, "Screen Capture", 
                               "Capture screenshots of full screen or specific regions",
                               self.run_screen_capture)
        
        self.create_tool_button(tools_frame, 1, "Window Info Display", 
                               "Monitor active window information and mouse position",
                               self.run_window_info_display)
        
        self.create_tool_button(tools_frame, 2, "Screen Overlay", 
                               "Display transparent overlay with grid for coordinate reference",
                               self.run_screen_overlay)
        
        self.create_tool_button(tools_frame, 3, "Apotris Analyzer", 
                               "Find and analyze 'Apotris PC' window for game coordinates",
                               self.run_apotris_analyzer)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status & Information", padding="10")
        status_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        
        # Status text area
        self.status_text = scrolledtext.ScrolledText(status_frame, height=8, wrap=tk.WORD)
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add initial status
        self.log_status("Application started successfully")
        self.log_status("All tools loaded and ready to use")
        
        # Bottom frame for additional controls
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        bottom_frame.columnconfigure(0, weight=1)
        
        # Help and info buttons
        help_button = ttk.Button(bottom_frame, text="Help", command=self.show_help)
        help_button.grid(row=0, column=0, padx=(0, 5))
        
        about_button = ttk.Button(bottom_frame, text="About", command=self.show_about)
        about_button.grid(row=0, column=1, padx=5)
        
        clear_log_button = ttk.Button(bottom_frame, text="Clear Log", command=self.clear_log)
        clear_log_button.grid(row=0, column=2, padx=5)
        
        # Exit button
        exit_button = ttk.Button(bottom_frame, text="Exit", command=self.root.quit)
        exit_button.grid(row=0, column=3, padx=(5, 0))
        
    def create_tool_button(self, parent, row, title, description, command):
        """Create a tool button with description"""
        # Tool button
        button = ttk.Button(parent, text=title, command=command, width=20)
        button.grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=2)
        
        # Description label
        desc_label = ttk.Label(parent, text=description, font=("Arial", 9))
        desc_label.grid(row=row, column=1, sticky=tk.W, pady=2)
        
    def log_status(self, message):
        """Log a message to the status area"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
        
    def clear_log(self):
        """Clear the status log"""
        self.status_text.delete(1.0, tk.END)
        self.log_status("Log cleared")
        
    def run_screen_capture(self):
        """Run the screen capture tool"""
        try:
            self.log_status("Starting Screen Capture tool...")
            
            # Import and run screen capture
            import pyautogui
            
            # Capture full screen
            screenshot = pyautogui.screenshot()
            screenshot.save("full_screen_capture.png")
            self.log_status("Full screen captured and saved as 'full_screen_capture.png'")
            
            # Capture a region
            region_screenshot = pyautogui.screenshot(region=(100, 100, 500, 300))
            region_screenshot.save("region_capture.png")
            self.log_status("Region captured and saved as 'region_capture.png'")
            
            messagebox.showinfo("Success", "Screen captures completed!\nCheck the generated PNG files.")
            
        except Exception as e:
            error_msg = f"Error in screen capture: {e}"
            self.log_status(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def run_window_info_display(self):
        """Run the window info display tool"""
        try:
            self.log_status("Starting Window Info Display tool...")
            
            # Run in a separate thread to avoid blocking the main UI
            def run_display():
                try:
                    create_gui()
                except Exception as e:
                    self.log_status(f"Error in window info display: {e}")
            
            thread = threading.Thread(target=run_display, daemon=True)
            thread.start()
            self.log_status("Window Info Display tool started in separate window")
            
        except Exception as e:
            error_msg = f"Error starting window info display: {e}"
            self.log_status(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def run_screen_overlay(self):
        """Run the screen overlay tool"""
        try:
            self.log_status("Starting Screen Overlay tool...")
            
            # Run in a separate thread
            def run_overlay():
                try:
                    from PyQt5.QtWidgets import QApplication
                    import sys
                    
                    app = QApplication(sys.argv)
                    from ScreenOverlay import MainWindow
                    window = MainWindow()
                    window.show()
                    app.exec_()
                except Exception as e:
                    self.log_status(f"Error in screen overlay: {e}")
            
            thread = threading.Thread(target=run_overlay, daemon=True)
            thread.start()
            self.log_status("Screen Overlay tool started")
            
        except Exception as e:
            error_msg = f"Error starting screen overlay: {e}"
            self.log_status(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def run_apotris_analyzer(self):
        """Run the Apotris analyzer tool"""
        try:
            self.log_status("Starting Apotris Analyzer tool...")
            
            # Run in a separate thread
            def run_analyzer():
                try:
                    analyzer = ApotrisAnalyzer()
                    result = analyzer.run_analysis()
                    if result:
                        self.log_status("Apotris analysis completed successfully")
                    else:
                        self.log_status("Apotris analysis failed - check console output")
                except Exception as e:
                    self.log_status(f"Error in Apotris analyzer: {e}")
            
            thread = threading.Thread(target=run_analyzer, daemon=True)
            thread.start()
            self.log_status("Apotris Analyzer tool started")
            
        except Exception as e:
            error_msg = f"Error starting Apotris analyzer: {e}"
            self.log_status(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def show_help(self):
        """Show help information"""
        help_text = """
Comp4010 AI Project - Screen Analysis Tools
==========================================

This application provides several tools for screen analysis and capture:

1. SCREEN CAPTURE
   - Captures full screen screenshots
   - Captures specific screen regions
   - Saves images as PNG files

2. WINDOW INFO DISPLAY
   - Monitors active window information
   - Tracks mouse position
   - Provides real-time window data
   - Keyboard shortcuts for mouse movement

3. SCREEN OVERLAY
   - Displays transparent grid overlay
   - Helps with coordinate reference
   - Click to close overlay

4. APOTRIS ANALYZER
   - Finds "Apotris PC" window
   - Analyzes game area surrounded by black pixels
   - Returns precise coordinates
   - Creates visualization images

REQUIREMENTS:
- Python 3.7+
- All dependencies from requirements.txt
- Windows OS (for win32gui functionality)

USAGE:
- Click any tool button to start that tool
- Check the status log for operation details
- Generated files will be saved in the current directory

For more information, check the individual Python files.
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        help_window.geometry("500x400")
        
        text_widget = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)
        
    def show_about(self):
        """Show about information"""
        about_text = """
Comp4010 AI Project
==================

Screen Analysis & Capture Tools

Author: Ryan M
Course: Comp4010 AI Project
Version: 1.0

This project provides comprehensive tools for:
- Screen capture and analysis
- Window monitoring and information display
- Coordinate detection and visualization
- Game area analysis for specific applications

Built with Python, OpenCV, PyQt5, and Windows API.

All tools are designed to work together for comprehensive
screen analysis and automation tasks.
        """
        
        messagebox.showinfo("About", about_text)
        
    def run(self):
        """Start the main application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.log_status("Application interrupted by user")
        except Exception as e:
            self.log_status(f"Unexpected error: {e}")
        finally:
            self.log_status("Application shutting down")

def main():
    """Main entry point"""
    print("Starting Comp4010 AI Project - Screen Analysis Tools")
    print("=" * 50)
    
    try:
        app = MainApplication()
        app.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return 1
    
    print("Application closed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
