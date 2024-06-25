import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
from PIL import Image
from ppocronnx import TextSystem

from globalInfo import GlobalInfo
from methodutil import split_actions
from statusModel import resnet34
from torchvision import transforms



class GetRewordUtil():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = resnet34(num_classes=12).to(self.device)
        self.load_model("models/resNet34_13.pth")

        self.class_indict = ["readInfo", "backHome", "killHero", "attackSmallDragon", "attackBigDragon", "attackEnemyCreeps",
                              "protectedOurSideCreeps", "attackEnemyMonster", "attackOurSideMonster", "attackEnemyTower", "protectedOurSideTower", "damageByTower"]

        # 全局状态
        self.globalInfo = GlobalInfo()

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

    def predict(self, img):
        if isinstance(img, np.ndarray):
            imgArr = Image.fromarray(img)

        data_transform = transforms.Compose([transforms.Resize(640),  # 验证过程图像预处理有变动，将原图片的长宽比固定不动，将其最小边长缩放到256
                                             transforms.CenterCrop(640),  # 再使用中心裁剪裁剪一个640×640大小的图片
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        imgArr = data_transform(imgArr)
        # expand batch dimension
        imgArr = torch.unsqueeze(imgArr, dim=0)

        # prediction
        self.model.eval()  # 使用eval模式
        with torch.no_grad():  # 不对损失梯度进行跟踪
            # predict class
            output = torch.squeeze(self.model(imgArr.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "class: {}   prob: {:.3}".format(self.class_indict[predict_cla],
                                                     predict[predict_cla].numpy())
        # print(print_res)
        res = self.class_indict[predict_cla]
        if res == "backHome":
            res = None
        return res

    def calculate_reword(self, status_name):
        rewordResult = 0
        gamePassTime = self.globalInfo.get_game_time_pass()

        if status_name == "readInfo":
            rewordResult = 0
        elif status_name == "killHero":
            rewordResult = random.choice([1, -1])
        elif status_name == "attackSmallDragon":
            # 2分钟
            if gamePassTime > 120:
                rewordResult = 1
            # 5分钟
            elif gamePassTime > 300:
                rewordResult = 2
            # 10分钟
            elif gamePassTime > 600:
                rewordResult = 3
            # 20分钟
            elif gamePassTime > 1200:
                rewordResult = 0
            # 30分钟
            elif gamePassTime > 1800:
                rewordResult = -1
            else:
                rewordResult = -2
        elif status_name == "attackBigDragon":
            # 2分钟
            if gamePassTime > 120:
                rewordResult = 1
            # 5分钟
            elif gamePassTime > 300:
                rewordResult = 2
            # 10分钟
            elif gamePassTime > 600:
                rewordResult = 3
            # 20分钟
            elif gamePassTime > 1200:
                rewordResult = 0
            # 30分钟
            elif gamePassTime > 1800:
                rewordResult = -1
            else:
                rewordResult = -10
        elif status_name == "attackEnemyCreeps":
            # 10分钟
            if gamePassTime > 600:
                rewordResult = 2
            # 20分钟
            elif gamePassTime > 1200:
                rewordResult = 1
            else:
                rewordResult = 3

        elif status_name == "protectedOurSideCreeps":
            rewordResult = -2
        elif status_name == "attackEnemyMonster":
            # 10分钟
            if gamePassTime > 600:
                rewordResult = 2
            # 20分钟
            elif gamePassTime > 1200:
                rewordResult = 1
            # 25分钟
            elif gamePassTime > 1500:
                rewordResult = -1
            # 30分钟
            elif gamePassTime > 1800:
                rewordResult = -1
            else:
                rewordResult = 1
        elif status_name == "attackOurSideMonster":
            # 10分钟
            if gamePassTime > 600:
                rewordResult = 2
            # 20分钟
            elif gamePassTime > 1200:
                rewordResult = 1
            else:
                rewordResult = 1
        elif status_name == "attackEnemyTower":
            # 5分钟
            if gamePassTime > 300:
                rewordResult = 2
            # 10分钟
            elif gamePassTime > 600:
                rewordResult = 1
            # 20分钟
            elif gamePassTime > 1200:
                rewordResult = 1
            else:
                rewordResult = 0
        elif status_name == "protectedOurSideTower":
            rewordResult = -2
        elif status_name == "damageByTower":
            rewordResult = -1
        elif status_name == "successes":
            rewordResult = 10000
        elif status_name == "failed":
            rewordResult = -10000
        elif status_name == "death":
            rewordResult = -1

        if self.globalInfo.is_back_home_over():
            action1_logits, angle1_logits, action2_logits, type2_logits, angle2_logits, duration2_logits = split_actions(self.globalInfo.get_value("action"))

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
