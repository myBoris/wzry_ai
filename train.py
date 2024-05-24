import threading
import time

import cv2
import numpy as np
import scrcpy

from androidController import AndroidController
from getReword import GetRewordUtil
from methodutil import count_parameters
from wzry_agent import Agent

from wzry_env import Environment

EPOCHS = 100

window_name = "wzry_ai"

agent = Agent(load_model=True)

# 打印模型的参数数量
count_parameters(agent.model)
rewordUtil = GetRewordUtil()

lock = threading.Lock()
# 全局变量声明
globalFrame = None


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


controller = AndroidController(run_scrcpy())
env = Environment(window_name, controller, rewordUtil)

return_list = []
epoch = 0
state = None
next_state = None
while True:
    done = -1
    episode_return = 0
    # state = screenshot(window_name)

    lock.acquire()
    try:
        if globalFrame is not None:
            state = globalFrame
            next_state = globalFrame
    finally:
        lock.release()

    if state is None:
        print('client frame is None')
        time.sleep(0.1)
        continue
    if next_state is None:
        print('client frame is None')
        time.sleep(0.1)
        continue

    start_log = 0
    # isDone ===-1 对局未开始
    # isDone ===0 对局开始
    # isDone ===1 对局结束
    while done == -1 or done == 0:
        print('client frame ok')

        action = agent.act(state)

        # print("action", action)

        env.step(action)

        lock.acquire()
        try:
            if globalFrame is not None:
                next_state = globalFrame
        finally:
            lock.release()

        if next_state is None:
            print('client frame is None')
            time.sleep(0.01)
            continue

        reward, done, info = env.get_reword(next_state)

        # print("get reword", reward, done, info)

        # 对局未开始
        if done == -1:
            print("对局未开始")
            continue

        if start_log == 0:
            print("对局开始")
            start_log = start_log + 1

        # 追加经验
        agent.remember(state, action, reward, next_state, done)

        state = next_state

        episode_return += reward

        # 对局开始
        if done == 1:
            print(f"Episode: {epoch}/{EPOCHS}, Reward: {episode_return},  Time: {time}, Epsilon: {agent.epsilon}")
            break

        agent.replay(epoch)

    print("对局结束")
    # 保存每一局结束的reword
    return_list.append(episode_return)
    # 计算前n个元素的平均值
    average = np.mean(return_list[:epoch])
    print("average reword", average)

    epoch = epoch + 1

    # agent.save(f"src/model_episode_{epoch}.pt")
