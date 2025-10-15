import sys

from PyQt5 import QtGui, QtCore, uic
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.X11BypassWindowManagerHint
        )
        self.setGeometry(
            QtWidgets.QStyle.alignedRect(
                QtCore.Qt.LeftToRight, QtCore.Qt.AlignCenter,
                QtCore.QSize(500, 600),
                QtWidgets.qApp.desktop().availableGeometry()
        ))
          # Set attribute for translucent background
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # Style the background to be transparent (important for effect)
        self.setStyleSheet("background: transparent;")

        layout = QVBoxLayout()
        label = QLabel("This text is opaque on a transparent background.")
        layout.addWidget(label)

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def paintEvent(self, event):
        """Draw grid pattern"""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Transparent background
        painter.fillRect(self.rect(), QtCore.Qt.transparent)

        # Grid color (semi-transparent white lines)
        grid_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 150))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)

        # Grid spacing in pixels
        spacing = 40

        # Draw vertical lines
        for x in range(0, self.width(), spacing):
            if x > 5:
                grid_pen.setColor(QtGui.QColor(255, 0, 0, 200))  # Change color for the first line
                grid_pen.setWidth(2)
                painter.setPen(grid_pen)
            painter.drawLine(x, 0, x, self.height())

        # Draw horizontal lines
        for y in range(0, self.height(), spacing):
            painter.drawLine(0, y, self.width(), y)

    def mousePressEvent(self, event):
        QtWidgets.qApp.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mywindow = MainWindow()
    mywindow.show()
    app.exec_()