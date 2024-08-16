import sys
import os
import json
import subprocess
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, \
    QSizePolicy, QFileDialog, QTextEdit, QLineEdit
from PyQt5.QtCore import pyqtSlot, QTimer, Qt
from PyQt5.QtGui import QPalette, QColor
from filelock import FileLock, Timeout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TrainingPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.training_process = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plots_from_file)
        self.timer.start(1000)  # 每秒更新一次

        # 初始化输出区域，并显示初始状态信息
        self.log_message("应用程序已启动。等待用户操作。")
        self.set_training_status("未开始")

    def initUI(self):
        main_layout = QVBoxLayout()

        # 顶部布局包含算法选择和控制按钮
        top_layout = QVBoxLayout()  # 将top_layout改为QVBoxLayout以便放置多个控件
        main_layout.addLayout(top_layout)

        # 上部分的算法选择和控制按钮
        control_layout = QHBoxLayout()
        algo_label = QLabel("选择算法:")
        control_layout.addWidget(algo_label)

        self.algo_combo = QComboBox()
        self.algo_combo.addItems(['PPO', 'TD3', 'DQN', 'A2C', 'SAC'])
        self.algo_combo.currentIndexChanged.connect(self.on_algo_changed)
        control_layout.addWidget(self.algo_combo)

        self.start_button = QPushButton("执行")
        self.start_button.clicked.connect(self.start_training)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("终止")
        self.stop_button.clicked.connect(self.stop_training)
        control_layout.addWidget(self.stop_button)

        # 清空输出区域按钮
        self.clear_output_button = QPushButton("清空输出")
        self.clear_output_button.clicked.connect(self.clear_output_area)
        control_layout.addWidget(self.clear_output_button)

        # 状态显示文本区域，设置固定宽度并根据状态改变背景颜色
        self.status_display = QLineEdit("未开始")
        self.status_display.setFixedWidth(100)  # 设置固定宽度为100px
        self.status_display.setReadOnly(True)
        self.set_training_status("未开始")  # 初始化状态为未开始
        control_layout.addWidget(self.status_display)

        top_layout.addLayout(control_layout)

        # 输出区域，设置固定高度
        self.output_area = QTextEdit()
        self.output_area.setFixedHeight(100)  # 设置固定高度为100px
        self.output_area.setReadOnly(True)  # 只读
        top_layout.addWidget(self.output_area)

        # 底部布局包含图表显示区域和控制按钮
        bottom_layout = QHBoxLayout()
        main_layout.addLayout(bottom_layout)

        # 增加左侧布局用于放置按钮，设置靠上排列
        button_widget = QWidget()
        button_layout = QVBoxLayout(button_widget)
        button_layout.setSpacing(10)
        button_layout.setAlignment(Qt.AlignTop)  # 设置按钮从上往下排列
        bottom_layout.addWidget(button_widget)

        self.load_json_button = QPushButton("读取JSON并显示")
        self.load_json_button.setFixedHeight(40)  # 设置固定高度
        self.load_json_button.clicked.connect(self.update_plots_from_file)
        button_layout.addWidget(self.load_json_button)

        self.clear_json_button = QPushButton("清空JSON数据")
        self.clear_json_button.setFixedHeight(40)  # 设置固定高度
        self.clear_json_button.clicked.connect(self.clear_json_data)
        button_layout.addWidget(self.clear_json_button)

        self.save_image_button = QPushButton("保存到图片")
        self.save_image_button.setFixedHeight(40)  # 设置固定高度
        self.save_image_button.clicked.connect(self.save_plot_to_image)
        button_layout.addWidget(self.save_image_button)

        # 图表部分
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        bottom_layout.addWidget(self.canvas)

        self.setLayout(main_layout)
        self.setWindowTitle('强化学习面板')
        self.setGeometry(100, 100, 1400, 900)  # 设置面板大小为1400x900

    def log_message(self, message):
        """打印状态信息到输出区域"""
        self.output_area.append(message)

    def clear_output_area(self):
        """清空输出区域"""
        self.output_area.clear()

    def set_training_status(self, status):
        """设置训练状态并更改背景颜色"""
        self.status_display.setText(status)

        # 根据状态设置背景颜色
        palette = self.status_display.palette()
        if status == "未开始":
            palette.setColor(QPalette.Base, QColor(Qt.white))
        elif status == "训练中":
            palette.setColor(QPalette.Base, QColor("#bad7df"))  # 背景色设置为灰色
        elif status == "训练结束":
            palette.setColor(QPalette.Base, QColor("#61c0bf"))  # 背景色设置为绿色
        self.status_display.setPalette(palette)

    @pyqtSlot()
    def start_training(self):
        # 在启动新的训练前停止上一个训练
        self.stop_training()

        # 清空 JSON 数据文件
        self.clear_json_data()

        algorithm = self.algo_combo.currentText()
        self.log_message(f"Starting training with {algorithm}...")
        self.set_training_status("训练中")

        # 根据算法选择不同的训练脚本
        script_map = {
            'PPO': 'train_ppo.py',
            'TD3': 'train_td3.py',
            'DQN': 'train_dqn.py',
            'A2C': 'train_a2c.py',
            'SAC': 'train_sac.py'
        }

        script = script_map.get(algorithm)
        if script:
            # 启动子进程，不重定向 stdout 和 stderr
            self.training_process = subprocess.Popen(
                [sys.executable, script]
            )

    @pyqtSlot()
    def stop_training(self):
        if self.training_process:
            self.training_process.terminate()
            self.training_process = None
            self.log_message("Training stopped.")
            self.set_training_status("训练结束")

    @pyqtSlot()
    def clear_json_data(self):
        lock = FileLock("training_data.json.lock", timeout=5)  # 设置锁定超时时间
        try:
            with lock:
                if os.path.exists('training_data.json'):
                    # 打开文件并写入空列表（清空文件内容）
                    with open('training_data.json', 'w') as file:
                        json.dump([], file, indent=4)

                    # 清除图表内容
                    self.figure.clear()
                    self.canvas.draw()

                    self.log_message("JSON数据已清空.")
        except Timeout:
            self.log_message("锁定文件超时，无法清空JSON数据。")

    @pyqtSlot()
    def save_plot_to_image(self):
        # 打开文件对话框以选择保存位置
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "PNG Files (*.png);;All Files (*)",
                                                   options=options)
        if file_path:
            # 保存图表到图片
            self.figure.savefig(file_path)
            self.log_message(f"图表已保存到 {file_path}")

    def update_plots_from_file(self):
        lock = FileLock("training_data.json.lock", timeout=5)  # 设置锁定超时时间
        try:
            with lock:
                if os.path.exists('training_data.json'):
                    with open('training_data.json', 'r') as file:
                        data = json.load(file)

                        self.figure.clear()
                        for i, plot_data in enumerate(data):
                            ax = self.figure.add_subplot(len(data), 1, i + 1)
                            x_data = plot_data['x_data']
                            y_data = plot_data['y_data']
                            title = plot_data['title']
                            description = plot_data['description']
                            x_label = plot_data.get('x_label', 'Episodes')
                            y_label = plot_data.get('y_label', 'Value')

                            ax.plot(x_data, y_data)
                            ax.set_title(f'{title}\n{description}')
                            ax.set_xlabel(x_label)
                            ax.set_ylabel(y_label)

                        # 自动调整子图布局，避免重叠
                        self.figure.tight_layout(pad=3.0)
                        self.canvas.draw()
        except Timeout:
            self.log_message("锁定文件超时，无法读取JSON数据。")

    @pyqtSlot()
    def on_algo_changed(self):
        """当算法选择变化时，重置状态为未开始"""
        self.set_training_status("未开始")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    panel = TrainingPanel()
    panel.show()
    sys.exit(app.exec_())
