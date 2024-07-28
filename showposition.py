import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox, QTextEdit
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QPoint
import subprocess
import os

class ImageWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('坐标查看器')

        self.layout = QVBoxLayout()

        # 显示图片的标签
        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.mousePressEvent = self.get_pos

        self.layout.addWidget(self.imageLabel)

        self.buttonLayout = QHBoxLayout()

        # 选择设备的标签和下拉列表
        self.deviceLabel = QLabel("选择设备:", self)
        self.buttonLayout.addWidget(self.deviceLabel)

        self.deviceComboBox = QComboBox(self)
        self.refresh_devices()
        self.buttonLayout.addWidget(self.deviceComboBox)

        self.buttonLayout.addStretch(1)  # 添加一个弹性空间

        # 打开图片按钮
        self.openButton = QPushButton('打开图片', self)
        self.openButton.clicked.connect(self.open_image)
        self.openButton.setStyleSheet("background-color: lightblue;")
        self.buttonLayout.addWidget(self.openButton)

        # 截取截图按钮
        self.screenshotButton = QPushButton('截取截图', self)
        self.screenshotButton.clicked.connect(self.capture_screenshot)
        self.screenshotButton.setStyleSheet("background-color: lightgreen;")
        self.buttonLayout.addWidget(self.screenshotButton)

        # 清空坐标信息按钮
        self.clearButton = QPushButton('清空坐标', self)
        self.clearButton.clicked.connect(self.clear_coordinates)
        self.clearButton.setStyleSheet("background-color: lightcoral;")
        self.buttonLayout.addWidget(self.clearButton)

        self.layout.addLayout(self.buttonLayout)

        # 显示坐标信息格式说明的标签
        self.coordLabel = QLabel("点击坐标格式: (X,Y) / (width,height) = (x_percent,y_percent)", self)
        self.layout.addWidget(self.coordLabel)

        # 显示坐标信息的文本框
        self.coordDisplay = QTextEdit(self)
        self.coordDisplay.setReadOnly(True)
        self.layout.addWidget(self.coordDisplay)

        self.setLayout(self.layout)

        # 如果不存在则创建 images 目录
        if not os.path.exists('images'):
            os.makedirs('images')

        # 打开默认图片如果存在
        default_image_path = 'images/screen.png'
        if os.path.exists(default_image_path):
            self.load_image(default_image_path)

    def open_image(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "打开图片文件", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if fileName:
            self.load_image(fileName)

    def load_image(self, file_path):
        self.pixmap = QPixmap(file_path)
        self.imageLabel.setPixmap(self.pixmap)
        self.imageLabel.resize(self.pixmap.width(), self.pixmap.height())

    def get_pos(self, event):
        x = event.pos().x()
        y = event.pos().y()
        width = self.imageLabel.width()
        height = self.imageLabel.height()
        x_percent = x / width
        y_percent = y / height
        coord_text = f'({x}, {y}) / ({width}, {height}) = ({x_percent:.3f}, {y_percent:.3f})'
        self.coordDisplay.append(coord_text)
        print(coord_text)

    def refresh_devices(self):
        result = subprocess.run(['adb', 'devices'], stdout=subprocess.PIPE)
        output = result.stdout.decode()
        devices = [line.split('\t')[0] for line in output.split('\n') if 'device' in line and not line.startswith('List')]
        self.deviceComboBox.clear()
        self.deviceComboBox.addItems(devices)

    def capture_screenshot(self):
        device = self.deviceComboBox.currentText()
        if device:
            local_image_path = 'images/screen.png'
            if os.path.exists(local_image_path):
                os.remove(local_image_path)
            subprocess.run(['adb', '-s', device, 'exec-out', 'screencap -p'], stdout=open(local_image_path, 'wb'))
            self.load_image(local_image_path)

    def clear_coordinates(self):
        self.coordDisplay.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageWidget()
    ex.show()
    sys.exit(app.exec_())
