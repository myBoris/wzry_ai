import threading
import time
import numpy as np
from android_tool import AndroidTool
from argparses import args
from dqnAgent import DQNAgent
from getReword import GetRewordUtil
from globalInfo import GlobalInfo

from wzry_env import Environment
from onnxRunner import OnnxRunner

# 全局状态
globalInfo = GlobalInfo()

class_names = ['started']
start_check = OnnxRunner('models/start.onnx', classes=class_names)

rewordUtil = GetRewordUtil()
tool = AndroidTool()
tool.show_scrcpy()
env = Environment(tool, rewordUtil)

agent = DQNAgent()

def data_collector():
    while True:
        # 获取当前的图像
        state = tool.screenshot_window()
        # 保证图像能正常获取
        if state is None:
            time.sleep(0.01)
            continue
        # 初始化对局状态 对局未开始
        globalInfo.set_game_end()
        # 判断对局是否开始
        checkGameStart = start_check.get_max_label(state)

        if checkGameStart == 'started':
            print("-------------------------------对局开始-----------------------------------")
            globalInfo.set_game_start()

            # 对局开始了，进行训练
            while globalInfo.is_start_game():
                # 获取预测动作
                action = agent.select_action(state)

                next_state, reward, done, info = env.step(action)
                print(info, reward)

                # 对局结束
                if done == 1:
                    print("-------------------------------对局结束-----------------------------------")
                    globalInfo.set_game_end()
                    break

                # 追加经验
                globalInfo.store_transition_dqn(state, action, reward, next_state, done)

                state = next_state

        else:
            print("对局未开始")
            time.sleep(0.1)


def train_agent():
    count = 1
    while True:
        if not globalInfo.is_memory_bigger_batch_size_dqn():
            time.sleep(1)
            continue
        print("training")
        agent.replay()
        if count % args.num_episodes == 0:
            agent.save_model('src/wzry_ai.pt')
        count = count + 1
        if count >= 100000:
            count = 1


if __name__ == '__main__':
    training_thread = threading.Thread(target=train_agent)
    training_thread.start()
    data_collector()
