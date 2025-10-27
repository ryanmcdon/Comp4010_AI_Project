import sys

from PyQt5 import QtGui, QtCore, uic
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout
import random

class ScreenOverlay(QtWidgets.QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.X11BypassWindowManagerHint
        )
          # Set attribute for translucent background
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

        # Initialize variables
        self.resize(111, 221)
        self.block_width = 10
        self.block_height = 20
        self.spacing = 11 # Spacing between blocks
        self.block_array = []
        self.grid_pixmap = None

        # Build the grid once
        self.generate_grid()
        self.updateDots()

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # TEST: Update this over and over
        '''
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateDots)
        self.timer.start(1000)
        '''
        # QtCore.QTimer.singleShot(2000, lambda: self.update_window_location(400, 600))

    def update_window_location(self, x, y):
        """Move the overlay window to a new location"""
        self.move(x, y)

    def generate_grid(self):
        """Initialize the static grid into a QPixmap"""
        pixmap = QtGui.QPixmap(self.size())
        pixmap.fill(QtCore.Qt.transparent)

        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        grid_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 100))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)

        for x in range(0, self.width(), self.spacing):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), self.spacing):
            painter.drawLine(0, y, self.width(), y)

        painter.end()
        self.grid_pixmap = pixmap

    # Update the dots visual (TEST)
    def updateDots(self):
        """Generate new random dots and refresh screen"""
        self.block_array.clear()
        for _ in range((self.block_width + 1) * (self.block_height + 1)):
            i = random.randint(0, 2)
            self.block_array.append(i)
        self.update()  # triggers paintEvent()

    def setDots(self, block_array):
        """Set the dot pattern based on external input"""
        self.block_array = block_array
        self.update()  # triggers paintEvent()

    # Paint the lines
    def paintEvent(self, event):
        """Draw grid pattern"""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Transparent background
        if self.grid_pixmap:
            painter.drawPixmap(0, 0, self.grid_pixmap)

        # Dot brushes
        red_brush = QtGui.QBrush(QtGui.QColor(255, 0, 0, 255))
        green_brush = QtGui.QBrush(QtGui.QColor(0, 255, 0, 255))
        painter.setBrush(red_brush)
        painter.setPen(QtCore.Qt.NoPen)

        """Draw dots based on block_array"""
        counter = 0
        # Changes colour based on block_array values (0 = no dot, 1 = red dot, 2 = green dot)
    
        for y in range(self.block_height):
            for x in range(self.block_width):
                if self.block_array[counter] == True:
                    painter.setBrush(red_brush)
                    painter.drawEllipse(QtCore.QPointF((x * self.spacing) + self.spacing/2, (y * self.spacing) + self.spacing/2), 2, 2)
                elif self.block_array[counter] == 2:
                    painter.setBrush(green_brush)
                    painter.drawEllipse(QtCore.QPointF((x * self.spacing) + self.spacing/2, (y * self.spacing) + self.spacing/2), 2, 2)
                counter += 1

    def mousePressEvent(self, event):
        QtWidgets.qApp.quit()

    pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mywindow = ScreenOverlay()
    mywindow.show()
    app.exec_()