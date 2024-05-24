from methodutil import  conver_model_result_to_action


class Environment():
    def __init__(self, android_controller, rewordUtil):
        self.android_controller = android_controller
        self.rewordUtil = rewordUtil

    def step(self, action):
        real_action = conver_model_result_to_action(action)
        self.android_controller.execute_actions(real_action)
        # next_state = screenshot(self.window_name)

    def get_reword(self, status):
        reward, done, info = self.rewordUtil.get_reword(status, True)
        return reward, done, info
