import time
import numpy as np
import scrcpy

import argparses
from androidController import AndroidController
from getReword import GetRewordUtil
from globalInfo import GlobalInfo
from methodutil import count_parameters
from wzry_agent import Agent

from wzry_env import Environment
from onnxRunner import OnnxRunner

# 全局状态
globalInfo = GlobalInfo()

class_names = ['started']
start_check = OnnxRunner('models/start.onnx', classes=class_names)

agent = Agent()
# 打印模型的参数数量
count_parameters(agent.model)

# 全局变量声明
globalInfo.set_global_frame(None)

# window_title = "CPH2309"

globalInfo.set_value("count", 0)


def on_client_frame(frame):
    globalInfo.set_global_frame(frame)


#
#     if frame is not None:
#         recordImgFlg = globalInfo.get_value("recordImg")
#         if recordImgFlg is not None and recordImgFlg:
#             count = globalInfo.get_value("count")
#             cv2.imwrite(f"tmp/img_{count}.jpg", frame)
#             count = count+1
#             globalInfo.set_value("count", count)


def run_scrcpy():
    device_id = argparses.args.device_id
    # device_id = "192.168.0.75:5555"
    max_width = 1080
    max_fps = 60
    bit_rate = 2000000000

    client = scrcpy.Client(device=device_id, max_width=max_width, max_fps=max_fps, bitrate=bit_rate)
    client.add_listener(scrcpy.EVENT_FRAME, on_client_frame)
    client.start(threaded=True)

    return client


rewordUtil = GetRewordUtil()
controller = AndroidController(run_scrcpy())
env = Environment(controller, rewordUtil)

return_list = []
epoch = 0
state = None
next_state = None

while True:
    # 获取当前的图像
    state = globalInfo.get_global_frame()
    # 保证图像能正常获取
    if state is None:
        time.sleep(0.01)
        continue
    # cv2.imwrite(f"tmp/img_0.jpg", state)
    # 初始化对局状态 对局未开始
    globalInfo.set_game_end()
    # 判断对局是否开始
    checkGameStart = start_check.get_max_label(state)

    if checkGameStart == 'started':
        print("-------------------------------对局开始-----------------------------------")
        globalInfo.set_game_start()
        globalInfo.set_start_game_time()

        # 这一局的总回报
        epoch_return_total = 0
        # 对局开始了，进行训练
        while globalInfo.is_start_game():
            # 获取预测动作
            action = agent.act(state)
            globalInfo.set_value("action", action)
            # 执行动作
            next_state, reward, done, info = env.step(action)
            print(reward, info)

            # 对局结束
            if done == 1:
                print("-------------------------------对局结束-----------------------------------")
                globalInfo.set_game_end()
                print(f"Episode: {epoch}, Reward total: {epoch_return_total},  Time: {time}, Epsilon: {agent.epsilon}")
                break

            # 追加经验
            agent.remember(state, action, reward, next_state, done)

            state = next_state

            epoch_return_total += reward

            agent.replay()

        # 保存每一局结束的reword
        return_list.append(epoch_return_total)
        # 计算前n个元素的平均值
        average = np.mean(return_list[:epoch])
        print("average reword", average)
        epoch = epoch + 1

    else:
        print("对局未开始")
        time.sleep(0.1)
