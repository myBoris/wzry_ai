import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
from PIL import Image

from statusModel import resnet34
from torchvision import transforms

from templateMatcher import TemplateMatcher


class GetRewordUtil():
    def __init__(self, templateMatcher=TemplateMatcher(threshold=0.8)):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = resnet34(num_classes=12).to(self.device)
        self.load_model("resNet34_13.pth")
        # 结合类别名称与行为
        self.CATEGORIES = {
            "beforeStart": self.handle_beforeStart,
            "started": self.handle_started,
            "finish": self.handle_finish,
            "successes": self.handle_successes,
            "failed": self.handle_failed,
            "readInfo": self.handle_readInfo,
            "death": self.handle_death,
            "backHome": self.handle_backHome,
            "killHero": self.handle_killHero,
            "attackSmallDragon": self.handle_attackSmallDragon,
            "attackBigDragon": self.handle_attackBigDragon,
            "attackEnemyCreeps": self.handle_attackEnemyCreeps,
            "protectedOurSideCreeps": self.handle_protectedOurSideCreeps,
            "attackEnemyMonster": self.handle_attackEnemyMonster,
            "attackOurSideMonster": self.handle_attackOurSideMonster,
            "attackEnemyTower": self.handle_attackEnemyTower,
            "protectedOurSideTower": self.handle_protectedOurSideTower,
            "damageByTower": self.handle_damageByTower
        }

        json_path = 'class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        self.class_indict = json.load(json_file)
        self.matcher = templateMatcher

    def load_model(self, model_path):
        # model = torch.hub.load(r"yolov9-main", 'custom', path=model_path, force_reload=True, source='local')
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

    def predict(self, model, img):
        results = model(img)
        max_conf = 0
        best_detection = None
        for detection in results.xyxy[0]:
            x1, y1, x2, y2, conf, class_id = detection.cpu().numpy()
            if conf > max_conf:
                max_conf = conf
                best_detection = detection

        if best_detection is not None:
            x1, y1, x2, y2, conf, class_id = best_detection.cpu().numpy()
            # print(f"Best Detection -> ID: {int(class_id)}, Name: {list(self.CATEGORIES.keys())[int(class_id)]}, Conf: {conf:.2f}, Box: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")
            return int(class_id), list(self.CATEGORIES.keys())[int(class_id)]
        else:
            # print(f"Best Detection -> ID: -1, Name: not found")
            return -1, "not found"
    def predict2(self, img):

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        data_transform = transforms.Compose([transforms.Resize(640),  # 验证过程图像预处理有变动，将原图片的长宽比固定不动，将其最小边长缩放到256
                                   transforms.CenterCrop(640),  # 再使用中心裁剪裁剪一个640×640大小的图片
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # prediction
        self.model.eval()  # 使用eval模式
        with torch.no_grad():  # 不对损失梯度进行跟踪
            # predict class
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "class: {}   prob: {:.3}".format(self.class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())
        # print(print_res)

        return self.class_indict[str(predict_cla)]



    # Define action functions
    # 攻击小兵
    def handle_beforeStart(self):
        return 0

    # 攻击小龙
    def handle_started(self):
        return 0

    # 攻击大龙
    def handle_finish(self):
        return 0

    # 击杀英雄
    def handle_successes(self):
        return 1000

    # 攻击敌方野怪
    def handle_failed(self):
        return -1000

    # 攻击己方野怪
    def handle_readInfo(self):
        return 0

    # 攻击塔
    def handle_death(self):
        return -0.1

    # 回城
    def handle_backHome(self):
        return 0

    # 被塔攻击
    def handle_killHero(self):
        return -0.5

    # 死亡
    def handle_attackSmallDragon(self):
        return 0.01

    # 查看信息
    def handle_attackBigDragon(self):
        return 0.01

    # 成功
    def handle_attackEnemyCreeps(self):
        return 0.01

    # 失败
    def handle_protectedOurSideCreeps(self):
        return 0.1

    # 开始
    def handle_attackEnemyMonster(self):
        return 0.01

    # 结束
    def handle_attackOurSideMonster(self):
        return 0.1

        # 结束

    def handle_attackEnemyTower(self):
        return 0.1

    def handle_protectedOurSideTower(self):
        return 0.1

    def handle_damageByTower(self):
        return -0.2

    def getModelStatus(self, image):
        class_name = self.predict2(image)
        return class_name

    def get_reword(self, image_path, isFrame):

        if isFrame:
            image = image_path
        else:
            image = cv2.imread(image_path)

        class_name = None
        # 使用 ThreadPoolExecutor 进行并行处理
        with ThreadPoolExecutor() as executor:
            # 记录开始时间
            start_time_class_name = time.time()
            start_time_md_class_name = time.time()

            # 提交任务
            future_class_name = executor.submit(self.matcher.match, image)
            future_md_class_name = executor.submit(self.getModelStatus, image)

            # 等待所有任务完成
            for future in as_completed([future_class_name, future_md_class_name]):
                end_time = time.time()
                if future == future_class_name:
                    class_name = future.result()
                    print(f"tp运行时间: {end_time - start_time_class_name:.3f} 秒")

                elif future == future_md_class_name:
                    md_class_name = future.result()
                    print(f"md运行时间: {end_time - start_time_md_class_name:.3f} 秒")

            # 如果 class_name 未定义，则使用 md_class_name
            if not class_name:
                class_name = md_class_name


        rewordCount = 0
        if class_name in self.CATEGORIES:
            rewordCount = self.CATEGORIES[class_name]()

        if class_name == 'successes' or class_name == 'failed':
            done = 2
        elif class_name == 'started' or class_name in self.CATEGORIES:
            done = 1
        else:
            done = 0

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
