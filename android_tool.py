import datetime
import math
import random
import subprocess
import sys
import threading
import time
from queue import Queue, Empty
import concurrent.futures

import cv2
import numpy as np
import win32gui
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication

from argparses import move_actions_detail, info_actions_detail, attack_actions_detail, args


class AndroidTool:
    def __init__(self, scrcpy_dir="scrcpy-win64-v2.0"):
        self.scrcpy_dir = scrcpy_dir
        self.device_serial = args.iphone_id
        self.actual_height, self.actual_width = self.get_device_resolution()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self._show_action_log = False

    def show_action_log(self):
        self._show_action_log = True

    def hidden_action_log(self):
        self._show_action_log = False

    def get_device_resolution(self):
        # 获取设备的实际分辨率
        output = subprocess.check_output(
            [f"{self.scrcpy_dir}/adb", "-s", self.device_serial, "shell", "wm", "size"]
        ).decode('utf-8')
        resolution = output.split()[-1].split('x')
        return int(resolution[0]), int(resolution[1])

    def execute_move(self, task_params):
        # 移动逻辑
        action_index = task_params['action']
        if action_index == 1:

            actions_detail = move_actions_detail[action_index]
            if self._show_action_log:
                print(actions_detail['action_name'])
            start_x, start_y = self.calculate_startpoint(actions_detail['position'])

            end_x, end_y = self.calculate_endpoint((start_x, start_y),
                                                   actions_detail['radius'],
                                                   task_params['angle'])

            subprocess.run([f"{self.scrcpy_dir}/adb", "-s", self.device_serial, "shell",
                            "input", "swipe", str(start_x), str(start_y), str(end_x), str(end_y), "500"])

    def execute_info(self, task_params):
        # 信息操作逻辑
        action_index = task_params['action']
        if not action_index == 0:
            actions_detail = info_actions_detail[action_index]
            if self._show_action_log:
                print(actions_detail['action_name'])
            start_x, start_y = self.calculate_startpoint(actions_detail['position'])
            subprocess.run([f"{self.scrcpy_dir}/adb", "-s", self.device_serial, "shell",
                            "input", "tap", str(start_x), str(start_y)])

    def execute_attack(self, task_params):
        # 攻击操作逻辑
        action_index = task_params['action']
        action_type = task_params['action_type']
        arg1 = task_params['arg1']
        arg2 = task_params['arg2'] + 1
        arg3 = task_params['arg3'] + 1

        if action_index != 0:
            actions_detail = attack_actions_detail[action_index]
            if self._show_action_log:
                print(actions_detail['action_name'])
            start_x, start_y = self.calculate_startpoint(actions_detail['position'])
            if action_index < 7:
                subprocess.run([f"{self.scrcpy_dir}/adb", "-s", self.device_serial, "shell",
                                "input", "tap", str(start_x), str(start_y)])
            else:
                if action_type == 0:
                    subprocess.run([f"{self.scrcpy_dir}/adb", "-s", self.device_serial, "shell",
                                    "input", "tap", str(start_x), str(start_y)])
                elif action_type == 1:
                    end_x, end_y = self.calculate_endpoint((start_x, start_y),
                                                           arg2,
                                                           arg1)
                    subprocess.run([f"{self.scrcpy_dir}/adb", "-s", self.device_serial, "shell",
                                    "input", "swipe", str(start_x), str(start_y), str(end_x), str(end_y), "500"])
                else:
                    subprocess.run([f"{self.scrcpy_dir}/adb", "-s", self.device_serial, "shell",
                                    "input", "swipe", str(start_x), str(start_y), str(start_x), str(start_y),
                                    str(arg3 * 1000)])

    def calculate_startpoint(self, center):
        p_x, p_y = center
        start_x = int(self.actual_width * p_x)
        start_y = int(self.actual_height * p_y)
        return start_x, start_y

    def calculate_endpoint(self, center, radius, angle):
        angle_rad = math.radians(angle)
        x = int(center[0] + radius * math.cos(angle_rad))
        y = int(center[1] + radius * math.sin(angle_rad))
        return x, y

    def show_scrcpy(self):
        subprocess.Popen(
            [f"{self.scrcpy_dir}/scrcpy.exe", "-s", self.device_serial, "-m", "1080", "--window-title",
             args.window_title])

    def action_move(self, params):
        self.executor.submit(self.execute_move, params)

    def action_attack(self, params):
        self.executor.submit(self.execute_attack, params)

    def action_info(self, params):
        self.executor.submit(self.execute_info, params)

    def stop(self):
        self.executor.shutdown(wait=True)

    def take_screenshot_save(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        screenshot_filename = f"screenshot_{timestamp}.png"

        try:
            result = subprocess.run([f'{self.scrcpy_dir}/adb', 'exec-out', 'screencap', '-p'], capture_output=True,
                                    text=False)

            if result.returncode == 0:
                with open(screenshot_filename, 'wb') as f:
                    f.write(result.stdout)
                print(f"Screenshot saved to {screenshot_filename}")
            else:
                print(f"Failed to take screenshot. Error: {result.stderr.decode('utf-8')}")
        except FileNotFoundError:
            print("adb is not installed or not found in your PATH.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def take_screenshot(self):
        try:
            result = subprocess.run([f'{self.scrcpy_dir}/adb', 'exec-out', 'screencap', '-p'], capture_output=True,
                                    text=False)

            if result.returncode == 0:
                screenshot_data = np.frombuffer(result.stdout, np.uint8)
                screenshot_image = cv2.imdecode(screenshot_data, cv2.IMREAD_COLOR)

                if screenshot_image is not None:
                    return screenshot_image
                else:
                    print("Failed to decode the screenshot.")
            else:
                print(f"Failed to take screenshot. Error: {result.stderr.decode('utf-8')}")
        except FileNotFoundError:
            print("adb is not installed or not found in your PATH.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return None

    def screenshot_window(self):
        """
        截取指定窗口的内容并返回图像数据。

        参数:
        window_name (str): 窗口标题的部分或全部字符串。

        返回:
        np.ndarray: 截图的图像数据，如果窗口未找到则返回 None。
        """
        try:
            # 获取窗口句柄
            handle = win32gui.FindWindow(None, args.window_title)
            if handle == 0:
                raise Exception(f"窗口 '{args.window_title}' 未找到。")

            # 初始化 QApplication
            app = QApplication(sys.argv)
            screen = QApplication.primaryScreen()

            # 截取指定窗口的内容
            img = screen.grabWindow(handle).toImage()

            # 将 QImage 转换为 numpy 数组
            img = img.convertToFormat(QImage.Format.Format_RGB32)
            width = img.width()
            height = img.height()
            ptr = img.bits()
            ptr.setsize(height * width * 4)
            arr = np.array(ptr).reshape(height, width, 4)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

            return arr
        except Exception as e:
            print(e)
            return None


def generate_random_number(n):
    return random.randint(0, n)


