import re
import sys

import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QIntValidator
from PyQt5.QtCore import QMutex
import scrcpy


class ImageWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.default_x = 1000
        self.default_y = 148
        self.initUI()
        self.initScrcpy()
        self.mutex = QMutex()

    def initUI(self):
        self.setWindowTitle('SCRCPY Stream')
        layout = QVBoxLayout(self)

        self.label = QLabel(self)
        layout.addWidget(self.label)

        self.lineEditX = QLineEdit(self)
        self.lineEditX.setValidator(QIntValidator(0, 10000))
        self.lineEditX.setText(f"{self.default_x}")
        layout.addWidget(self.lineEditX)

        self.lineEditY = QLineEdit(self)
        self.lineEditY.setValidator(QIntValidator(0, 10000))
        self.lineEditY.setText(f"{self.default_y}")
        layout.addWidget(self.lineEditY)

        self.setLayout(layout)

    def initScrcpy(self):
        self.client = scrcpy.Client(device="528e7355", max_width=1080, max_fps=120, bitrate=2000000000)
        self.client.add_listener(scrcpy.EVENT_FRAME, self.get_frame)
        self.client.start(threaded=True)

    def get_frame(self, frame):
        if frame is not None:
            height, width, channel = frame.shape
            bytesPerLine = 3 * width

            # Try to lock the mutex
            if self.mutex.tryLock(3000):  # Try to lock for up to 3 seconds
                x = self.extract_number(self.lineEditX.text())
                y = self.extract_number(self.lineEditY.text())
                self.mutex.unlock()  # Release lock immediately after reading
            else:
                # If lock not obtained within 3 seconds, use default coordinates
                x = 100
                y = 100

            # Draw a red circle at the obtained coordinates
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)  # BGR format, red
            image = QImage(frame.data, width, height, bytesPerLine, QImage.Format_BGR888)
            self.label.setFixedSize(width, height)
            self.label.setPixmap(QPixmap.fromImage(image))

    def extract_number(self, text):
        match = re.search(r'\d+', text)
        if match:
            return int(match.group(0))
        return 100


# 创建 PyQt 应用
app = QApplication(sys.argv)
ex = ImageWindow()
ex.show()
sys.exit(app.exec_())
