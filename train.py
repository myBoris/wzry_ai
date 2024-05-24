import threading
import time

import cv2
import numpy as np
import scrcpy

from androidController import AndroidController
from getReword import GetRewordUtil
from globalInfo import GlobalInfo
from methodutil import count_parameters
from templateMatcher import TemplateMatcher
from wzry_agent import Agent

from wzry_env import Environment

agent = Agent(load_model=True)
# 打印模型的参数数量
count_parameters(agent.model)

templateMatcher = TemplateMatcher(threshold=0.8)
rewordUtil = GetRewordUtil(templateMatcher)

lock = threading.Lock()
# 全局变量声明
globalFrame = None


def on_client_frame(frame):
    global globalFrame
    lock.acquire()
    try:
        globalFrame = frame
    finally:
        lock.release()


def run_scrcpy():
    device_id = "528e7355"
    max_width = 1080
    max_fps = 60
    bit_rate = 2000000000

    client = scrcpy.Client(device=device_id, max_width=max_width, max_fps=max_fps, bitrate=bit_rate)
    client.add_listener(scrcpy.EVENT_FRAME, on_client_frame)
    client.start(threaded=True)

    return client


controller = AndroidController(run_scrcpy())
env = Environment(controller, rewordUtil)

return_list = []
epoch = 0
state = None
next_state = None

# 全局状态
globalInfo = GlobalInfo()

while True:
    # 获取当前的图像
    lock.acquire()
    try:
        state = globalFrame
    finally:
        lock.release()
    # 保证图像能正常获取
    if state is None:
        time.sleep(0.1)
        continue

    # 初始化对局状态 对局未开始
    globalInfo.set_game_end()
    # 判断对局是否开始
    checkGameStart = templateMatcher.match(state)
    if checkGameStart == 'started':
        print("-------------------------------对局开始-----------------------------------")
        globalInfo.set_game_start()

        # 这一局的总回报
        epoch_return_total = 0
        # 对局开始了，进行训练
        while globalInfo.is_start_game():
            print("checkGameStart", globalInfo.is_start_game())
            # 获取预测动作
            action = agent.act(state)
            # 执行动作
            env.step(action)

            lock.acquire()
            try:
                next_state = globalFrame
            finally:
                lock.release()

            if next_state is None:
                time.sleep(0.01)
                continue

            reward, done, info = env.get_reword(next_state)

            # 对局结束
            if done == 2:
                print("-------------------------------对局结束-----------------------------------")
                globalInfo.set_game_end()
                print(f"Episode: {epoch}, Reward total: {epoch_return_total},  Time: {time}, Epsilon: {agent.epsilon}")
                break

            # 追加经验
            agent.remember(state, action, reward, next_state, done)

            state = next_state

            epoch_return_total += reward

            agent.replay(epoch)

        # 保存每一局结束的reword
        return_list.append(epoch_return_total)
        # 计算前n个元素的平均值
        average = np.mean(return_list[:epoch])
        print("average reword", average)
        epoch = epoch + 1

    else:
        print("对局未开始")
