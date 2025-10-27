import sys

from PyQt5 import QtGui, QtCore, uic
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout
import random

class MainWindow(QMainWindow):
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
        self.resize(110, 220)
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

        # QtCore.QTimer.singleShot(2000, lambda: self.update_window_location(400, 600))

        # TEST: Update this over and over
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateDots)
        self.timer.start(1000)

    def update_window_location(self, x, y):
        self.move(x, y)

    def generate_grid(self):
        """Render the static grid into a QPixmap once"""
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

    # Update the dots visual
    def updateDots(self):
        """Generate new random dots and refresh screen"""
        self.block_array.clear()
        for _ in range(self.block_width * self.block_height + 1):
            i = random.randint(0, 2)
            self.block_array.append(i)
        self.update()  # triggers paintEvent()

    # Paint the lines
    def paintEvent(self, event):
        """Draw grid pattern"""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Transparent background
        if self.grid_pixmap:
            painter.drawPixmap(0, 0, self.grid_pixmap)

        # Draw dots
        dot_brush = QtGui.QBrush(QtGui.QColor(255, 0, 0, 255))
        painter.setBrush(dot_brush)
        painter.setPen(QtCore.Qt.NoPen)

        counter = 0

        for x in range(self.block_width):
            for y in range(self.block_height):
                counter += 1
                if self.block_array[counter] == 1:
                    painter.drawEllipse(QtCore.QPointF((x * self.spacing) - self.spacing/2, (y * self.spacing) - self.spacing/2), 2, 2)

    def mousePressEvent(self, event):
        QtWidgets.qApp.quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mywindow = MainWindow()
    mywindow.show()
    app.exec_()