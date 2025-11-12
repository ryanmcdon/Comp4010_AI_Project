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
from PyQt5 import QtWidgets

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ScreenCapture import *
    from ScreenInfoDisplay import *
    from ScreenOverlay import ScreenOverlay
    from ApotrisAnalyzer import ApotrisAnalyzer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required dependencies are installed:")
    print("pip install -r requirements.txt")

class MainApplication:

    def run(self):
        app = QtWidgets.QApplication(sys.argv)
        
        self.ScreenOverlay = ScreenOverlay()
        self.ScreenOverlay.show()

        self.apotris_analyzer = ApotrisAnalyzer()
        info = self.apotris_analyzer.run_analysis_tester()
        print(info['game_coordinates'])
        print(info['white_board'])
        print(info['contour'])
        print(info['board'])

        # Assign the extracted information to the overlay for display
        # self.ScreenOverlay.update_window_location(info['game_coordinates']['top_left'][0], info['game_coordinates']['top_left'][1])
        self.ScreenOverlay.update_window_location(1000, 800)  # Example fixed position
        print(info['game_coordinates']['top_left'][0], info['game_coordinates']['top_left'][1])

        self.ScreenOverlay.setDots(info['board'])


        sys.exit(app.exec_())

        
        
if __name__ == "__main__":
    # Example usage: analyzer = ApotrisAnalyzer(); analyzer.run_analysis()
    main = MainApplication()
    main.run()
