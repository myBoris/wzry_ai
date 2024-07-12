import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
from ppocronnx import TextSystem

from globalInfo import GlobalInfo
from methodutil import split_actions
from templateMatch import TemplateMatcher


class GetRewordUtil:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        template_dir = "template"
        self.matcher = TemplateMatcher(template_dir)
        # 全局状态
        self.globalInfo = GlobalInfo()

    def predict(self, img):
        is_attack, rewordCount = self.calculate_attack_reword(img)
        return is_attack, rewordCount

    def calculate_attack_reword(self, img):

        # 获取图像的尺寸
        image_height, image_width = img.shape[:2]

        # 截取矩形的固定宽度和高度
        width = image_width * 0.116
        height = image_height * 0.024

        total_area = int(width * height)

        # 计算中心顶部矩形的起始点
        # left = int(image_width * 0.568)
        left = int(image_width * 0.57)
        top = int(image_height * 0.019)  # 从顶部开始
        right = int(left + width)
        bottom = int(top + height)

        # 根据计算出的坐标裁剪图像
        cropped_img = img[top:bottom, left:right]

        # 将图片从BGR转换到HSV色彩空间
        hsv_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

        # 定义BGR颜色 #AF363E
        bgr_color = np.uint8([[[62, 54, 175]]])  # 注意这里是BGR格式
        hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)
        hue = hsv_color[0][0][0]

        # 设置颜色范围的容错率
        tolerance = 10  # 容差值可以根据需要调整

        # 定义HSV中想要提取的颜色范围
        lower_bound = np.array([hue - tolerance, 50, 50])
        upper_bound = np.array([hue + tolerance, 255, 255])

        # 使用cv2.inRange()函数找到图像中颜色在指定范围内的区域
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # 将掩码应用于原图像，只保留指定颜色的区域
        color_segment = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)

        # 转成灰度图
        gray = cv2.cvtColor(color_segment, cv2.COLOR_BGR2GRAY)

        # 找到指定颜色的最右边的位置
        rightmost_position = 0
        for col in range(gray.shape[1]):
            if np.any(gray[:, col] != 0):
                rightmost_position = col

        # 计算指定颜色的面积
        area = rightmost_position * height

        isAttack = False
        res = 0
        if area > 0:
            isAttack = True
            res = 10 - int((area * 10) / total_area)

        return isAttack, res

    def calculate_reword(self, status_name, attack_reword):
        rewordResult = 0
        gamePassTime = self.globalInfo.get_game_time_pass()

        if status_name is None:
            rewordResult = 0
        elif status_name == "attack":
            action1_logits, angle1_logits, action2_logits, type2_logits, angle2_logits, duration2_logits = split_actions(
                self.globalInfo.get_value("action"))

            # 左手的操作
            # 获取最可能的action
            action1 = torch.argmax(action1_logits, dim=1)  # 得到0-3之间的整数

            # 右手的操作
            # 获取最可能的action
            action2 = torch.argmax(action2_logits, dim=1)  # 得到0-20之间的整数

            attack_action = [3,4,5,6,10,11,12,13]
            if action1 != 0 or action2 in attack_action:

                rewordResult = attack_reword
            else:
                rewordResult = -1
        elif status_name == "backHome":
            action1_logits, angle1_logits, action2_logits, type2_logits, angle2_logits, duration2_logits = split_actions(
                self.globalInfo.get_value("action"))

            # 左手的操作
            # 获取最可能的action
            action1 = torch.argmax(action1_logits, dim=1)  # 得到0-3之间的整数

            # 右手的操作
            # 获取最可能的action
            action2 = torch.argmax(action2_logits, dim=1)  # 得到0-20之间的整数

            if action1 != 0 and action2 != 0:
                rewordResult = 1
            else:
                rewordResult = -1
        elif status_name == "successes":
            rewordResult = 10000
        elif status_name == "failed":
            rewordResult = -10000
        elif status_name == "death":
            rewordResult = -1

        return rewordResult

    def check_finish(self, image):
        text_sys = TextSystem()
        res = text_sys.detect_and_ocr(image)
        done = 0
        class_name = None
        for boxed_result in res:
            # print("{}, {:.3f}".format(boxed_result.ocr_text, boxed_result.score))
            if boxed_result.ocr_text == "胜利" or boxed_result.ocr_text == "VICTORY":
                done = 1
                class_name = 'successes'
                break
            elif boxed_result.ocr_text == "失败" or boxed_result.ocr_text == "DEFEAT":
                done = 1
                class_name = 'failed'
                break
        return done, class_name

    def get_reword(self, image_path, isFrame):
        if isFrame:
            image = image_path
        else:
            image = cv2.imread(image_path)

        done = 0
        class_name = None
        md_class_name = None
        # 使用 ThreadPoolExecutor 进行并行处理
        with ThreadPoolExecutor() as executor:
            # 记录开始时间
            start_time_class_name = time.time()
            start_time_md_class_name = time.time()

            # 提交任务,预测状态
            future_class_name = executor.submit(self.check_finish, image)
            future_md_class_name = executor.submit(self.predict, image)

            # 等待所有任务完成
            for future in as_completed([future_class_name, future_md_class_name]):
                end_time = time.time()
                if future == future_class_name:
                    done, class_name = future.result()
                    # print(f"tp运行时间: {end_time - start_time_class_name:.3f} 秒")

                elif future == future_md_class_name:
                    is_attack, attack_rewordCount = future.result()
                    if is_attack:
                        md_class_name = "attack"
                    # print(f"md运行时间: {end_time - start_time_md_class_name:.3f} 秒")

            # 如果没结束，判断局内状态
            if done == 0:
                class_name = md_class_name

        # 计算回报
        rewordCount = self.calculate_reword(class_name, attack_rewordCount)

        return rewordCount, done, class_name


if __name__ == '__main__':
    # rewordUtil = RewordUtil()
    # reword = rewordUtil.get_reword("../dataset/test3.png")
    # print("reword result", reword)

    # 实例化视频对象
    cap = cv2.VideoCapture("H:\\AI\\work\\wzry_status\\video\\video10.mp4")

    rewordUtil = GetRewordUtil()
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        # 如果帧读取正确，ret 为 True
        if not ret:
            print('无法收到视频帧数据（该视频流是否已结束？），程序正在退出')
            break

        # if count < 5:
        #     count = count + 1
        #     continue
        # else:
        #     count = 0

        # 显示该帧
        cv2.imshow('frame0', frame)

        reword = rewordUtil.get_reword(frame, True)
        print(reword)

        # time.sleep(0.1)

        # 当按下键盘 q 时，退出程序
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
