import threading
import time

import cv2
import scrcpy
import torch
import torchvision.transforms as transforms

import methodutil
from androidController import AndroidController
from model import WzryNet


def load_and_preprocess_image(image, model_input_size=(640, 640)):

    # 将图片从BGR转换为RGB
    # images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

    # 缩放图片到指定大小（仿照YOLO的方式，这里使用双线性插值）
    # image = cv2.resize(image, model_input_size, interpolation=cv2.INTER_LINEAR)

    # 将数据类型转换为float32并归一化像素值
    image = image.astype('float32') / 255.0

    # 转换为torch张量，并添加一个batch维度
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)  # 增加batch维度，因为模型期望批处理输入

    return image

# 全局变量声明
globalFrame = None
lock = threading.Lock()

def on_client_frame(frame):
    global globalFrame
    if frame is not None:
        # 将帧数据转换为 OpenCV 格式
        # np_frame = np.frombuffer(frame, np.uint8)
        # img = cv2.imdecode(np_frame, cv2.IMREAD_ANYCOLOR)
        # if frame is not None:
        lock.acquire()
        try:
            globalFrame = frame
            # print("图像解码成功!")
        finally:
            lock.release()

    else:
        # print('client frame is None')
        # print("图像解码失败!")
        pass

def run_scrcpy():
    device_id = "528e7355"
    max_width = 1080
    max_fps = 60
    bit_rate = 2000000000

    client = scrcpy.Client(device=device_id, max_width=max_width, max_fps=max_fps, bitrate=bit_rate)
    client.add_listener(scrcpy.EVENT_FRAME, on_client_frame)
    client.start(threaded=True)

    return client




# 使用示例
if __name__ == "__main__":
    # 示例使用
    controller = AndroidController(run_scrcpy())

    try:
        # 模型
        model = WzryNet()
        model.load_state_dict(torch.load("src/wzry_ai.pt"), strict=False)
        model.eval()  # 设置为评估模式

        # model = WzryNet()
        # model.eval()  # 设置为评估模式
        image = None

        while(True):
            # 记录开始时间
            start_time = time.time()

            lock.acquire()
            try:
                if globalFrame is not None:
                    image = globalFrame
            finally:
                lock.release()

            if image is None:
                print('client frame is None')
                time.sleep(0.1)
                continue
            processed_image = load_and_preprocess_image(image, model_input_size=(640, 640))

            # 记录并打印这一步骤的时间
            step1_time = time.time()
            print(f"第一步运行时间: {step1_time - start_time:.3f} 秒")

            with torch.no_grad():
                action = model(processed_image)
                # 记录并打印这一步骤的时间
                step2_time = time.time()
                print(f"第二步运行时间: {step2_time - step1_time:.3f} 秒")

                actions = methodutil.conver_model_result_to_action(action)
                # 记录并打印这一步骤的时间
                step3_time = time.time()
                print(f"第三步运行时间: {step3_time - step2_time:.3f} 秒")
                print(actions)

            controller.execute_actions(actions)
            time.sleep(0.5)  # 保持程序运行以观察效果
            # 打印总运行时间
            total_time = time.time() - start_time
            print(f"总运行时间: {total_time:.3f} 秒")
    finally:
        pass
