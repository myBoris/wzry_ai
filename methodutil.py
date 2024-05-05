import sys

import cv2
import win32gui
from PyQt5.QtWidgets import QApplication

def screenshot(window_name):
    """获取屏幕截图并返回"""

    handle = win32gui.FindWindow(None, window_name)
    app = QApplication(sys.argv)
    screen = QApplication.primaryScreen()
    img = screen.grabWindow(handle).toImage()
    img.save("images/temp.png")

    # 使用OpenCV加载原始图像
    image = cv2.imread("images/temp.png")
    return image