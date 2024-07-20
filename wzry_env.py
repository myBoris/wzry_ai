import itertools
import threading
import time

import torch

from globalInfo import GlobalInfo


class Environment():
    def __init__(self, android_controller, rewordUtil):
        self.android_controller = android_controller
        self.rewordUtil = rewordUtil
        self.globalInfo = GlobalInfo()
        self.lock = threading.Lock()

        # 输出，0 或 1
        move_action_list = list(range(2))
        # 输出，0 到 359
        angle_list = list(range(360))
        # 输出，0 到 8
        info_action_list = list(range(9))
        # 输出0-10
        attack_action_list = list(range(11))
        # 输出0-2
        action_type_list = list(range(3))
        # 输出0-356
        arg1_list = list(range(360))
        # 输出0-99
        arg2_list = list(range(100))
        # 输出0-4
        arg3_list = list(range(5))

        # 计算每个列表的长度
        lengths = [
            len(move_action_list),
            len(angle_list),
            len(info_action_list),
            len(attack_action_list),
            len(action_type_list),
            len(arg1_list),
            len(arg2_list),
            len(arg3_list)
        ]

        # 操作空间
        self.action_space_n = sum(lengths)

        print(self.action_space_n)

    def step(self, action):
        move_action, angle, info_action, attack_action, action_type, arg1, arg2, arg3 = action
        self.android_controller.action_move({"action": move_action, "angle": angle})
        self.android_controller.action_info({"action": info_action})
        self.android_controller.action_attack(
            {"action": attack_action, "action_type": action_type, "arg1": arg1, "arg2": arg2, "arg3": arg3})

        next_state = self.android_controller.screenshot_window()
        while next_state is None or next_state.size == 0:
            time.sleep(0.01)
            next_state = self.android_controller.screenshot_window()
            continue

        reward, done, info = self.rewordUtil.get_reword(next_state, True, (
            move_action, angle, info_action, attack_action, action_type, arg1, arg2, arg3))

        return next_state, reward, done, info


if __name__ == '__main__':
    Environment(None, None)
