import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
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
        matched_result = self.matcher.match_template(img)

    def calculate_reword(self, status_name):
        rewordResult = 0
        gamePassTime = self.globalInfo.get_game_time_pass()
        if status_name is None:
            rewordResult = 0
        elif "attackMonster" in status_name:
            rewordResult = 1
        elif status_name == "successes":
            rewordResult = 10000
        elif status_name == "failed":
            rewordResult = -10000
        elif status_name == "death":
            rewordResult = -1

        if self.globalInfo.is_back_home_over():
            action1_logits, angle1_logits, action2_logits, type2_logits, angle2_logits, duration2_logits = split_actions(
                self.globalInfo.get_value("action"))

            # 左手的操作
            # 获取最可能的action
            action1 = torch.argmax(action1_logits, dim=1)  # 得到0-3之间的整数

            # 右手的操作
            # 获取最可能的action
            action2 = torch.argmax(action2_logits, dim=1)  # 得到0-20之间的整数

            if action1 != 0 and action2 != 0 and action2 != 0:
                rewordResult = -1
            else:
                rewordResult = 0
        else:
            if status_name == "backHome":
                rewordResult = 0
                self.globalInfo.set_back_home_time()

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
                    md_class_name = future.result()
                    # print(f"md运行时间: {end_time - start_time_md_class_name:.3f} 秒")

            # 如果没结束，判断局内状态
            if done == 0:
                class_name = md_class_name

        # 计算回报
        rewordCount = self.calculate_reword(class_name)

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
