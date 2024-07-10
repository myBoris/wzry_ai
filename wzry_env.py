import threading
import time

from globalInfo import GlobalInfo
from methodutil import conver_model_result_to_action


class Environment():
    def __init__(self, android_controller, rewordUtil):
        self.android_controller = android_controller
        self.rewordUtil = rewordUtil
        self.globalInfo = GlobalInfo()
        self.lock = threading.Lock()

    def step(self, action):
        real_action = conver_model_result_to_action(action)
        self.android_controller.execute_actions(real_action)

        next_state = self.globalInfo.get_global_frame()
        while next_state is None:
            time.sleep(0.01)
            next_state = self.globalInfo.get_global_frame()
            continue

        reward, done, info = self.rewordUtil.get_reword(next_state, True)

        return next_state, reward, done, info
