import random
import subprocess
import threading
import time
import math
import scrcpy
import os


class AndroidController:
    def __init__(self, client):
        # 初始化客户端，设置最大尺寸和比特率
        self.thread = None
        self.client = client

        self.lock = threading.Lock()  # 线程锁，用于同步操作
        self.active_touches = {}  # 活跃的触摸事件字典，键为位置，值为线程和时间戳
        # 预设的动作字典，每个 action 对应一个屏幕坐标和半径
        self.actions = {
            'none': {'action_name': '无操作', 'position': (0, 0), 'radius': 0},
            'move': {'action_name': '移动', 'position': (178, 382), 'radius': 150},
            'attack': {'action_name': '攻击', 'position': (912, 416), 'radius': 0},
            'attack_pawn': {'action_name': '攻击小兵', 'position': (838, 444), 'radius': 0},
            'attack_tower': {'action_name': '攻击塔', 'position': (950, 346), 'radius': 0},
            'back_base': {'action_name': '回城', 'position': (560, 436), 'radius': 0},
            'restore_health': {'action_name': '恢复', 'position': (626, 436), 'radius': 0},
            'skill': {'action_name': '召唤师技能', 'position': (692, 436), 'radius': 0},
            'skill_equipment': {'action_name': '装备技能', 'position': (906, 190), 'radius': 0},
            'skill_1': {'action_name': '1技能', 'position': (768, 430), 'radius': 100},
            'skill_2': {'action_name': '2技能', 'position': (820, 336), 'radius': 100},
            'skill_3': {'action_name': '3技能', 'position': (910, 284), 'radius': 100},
            'skill_4': {'action_name': '4技能', 'position': (0, 0), 'radius': 100},
            'add_skill_1': {'action_name': '升级1技能', 'position': (721, 377), 'radius': 0},
            'add_skill_2': {'action_name': '升级2技能', 'position': (774, 288), 'radius': 0},
            'add_skill_3': {'action_name': '升级3技能', 'position': (864, 236), 'radius': 0},
            'add_skill_4': {'action_name': '升级4技能', 'position': (0, 0), 'radius': 0},
            'buy_equipment_1': {'action_name': '购买装备1', 'position': (142, 194), 'radius': 0},
            'buy_equipment_2': {'action_name': '购买装备2', 'position': (142, 248), 'radius': 0},
            'attention_1': {'action_name': '发起进攻', 'position': (1000, 68), 'radius': 0},
            'attention_2': {'action_name': '开始撤退', 'position': (1000, 108), 'radius': 0},
            'attention_3': {'action_name': '请求集合', 'position': (1000, 148), 'radius': 0}
        }

    def handle_touch_down(self, pos, timeout):
        # 处理触摸按下事件
        if pos in self.active_touches:
            # 如果同一位置已有活跃的触摸，先释放它
            self.client.control.touch(int(pos[0]), int(pos[1]), scrcpy.ACTION_UP)
            self.active_touches[pos][0].cancel()  # 取消旧的定时器线程
        timer = threading.Timer(timeout, self.auto_release_touch, [pos])
        self.client.control.touch(int(pos[0]), int(pos[1]), scrcpy.ACTION_DOWN)
        timer.start()
        self.active_touches[pos] = (timer, time.time())  # 记录新的触摸事件和启动定时器

    def auto_release_touch(self, pos):
        # 自动释放触摸
        if pos in self.active_touches:
            self.client.control.touch(int(pos[0]), int(pos[1]), scrcpy.ACTION_UP)
            del self.active_touches[pos]

    def execute_actions(self, actions):
        # 同时执行多个指令
        self.lock.acquire()
        try:
            for action in actions:
                data = self.actions.get(action['action'])  # 从预设中获取位置和半径
                if data:
                    pos = data['position']
                    if action['type'] == 'down':
                        timeout = action.get('timeout', 5)  # 如果没有指定超时时间，默认为5秒
                        self.handle_touch_down(pos, timeout)
                    elif action['type'] == 'up':
                        if pos in self.active_touches:
                            self.client.control.touch(int(pos[0]), int(pos[1]), scrcpy.ACTION_UP)
                            self.active_touches[pos][0].cancel()  # 取消定时器线程
                            del self.active_touches[pos]
                    elif action['type'] == 'swipe':
                        angle = action['angle']
                        radius = data['radius']
                        end_pos = self.calculate_endpoint(pos, radius, angle)
                        self.client.control.swipe(int(pos[0]), int(pos[1]), int(end_pos[0]), int(end_pos[1]), 0.5)
                    elif action['type'] == 'long_press':
                        self.client.control.touch(int(pos[0]), int(pos[1]), scrcpy.ACTION_DOWN)
                        time.sleep(float(action.get('duration', 1)))  # 长按时间
                        self.client.control.touch(int(pos[0]), int(pos[1]), scrcpy.ACTION_UP)
                    elif action['type'] == 'click':
                        self.client.control.touch(int(pos[0]), int(pos[1]), scrcpy.ACTION_DOWN)
                        time.sleep(0.1)  # 短暂延迟以模拟真实点击
                        self.client.control.touch(int(pos[0]), int(pos[1]), scrcpy.ACTION_UP)
        finally:
            self.lock.release()

    def calculate_endpoint(self, center, radius, angle):
        """
        计算基于圆心、半径和角度的终点坐标。

        参数:
            center (tuple): 圆心坐标 (x, y)，通常是滑动开始的位置。
            radius (int): 从圆心到终点的距离。
            angle (int): 从x轴正方向顺时针旋转的角度，单位是度。

        返回:
            tuple: 终点坐标 (x, y)。

        坐标系说明:
            - 0度从x轴正方向开始（图形界面中，水平向右是x轴的正方向）。
            - 角度沿顺时针方向增加。
            - 90度位于y轴负方向（图形界面中，垂直向上是y轴的负方向）。
            - 180度位于x轴负方向（向左）。
            - 270度位于y轴正方向（图形界面中，垂直向下是y轴的正方向）。

        示例:
            为了计算从点 (100, 200) 开始，半径为 100，角度为 90度的终点位置：
            起始点为 x轴正方向，顺时针旋转 90度，将会指向屏幕的上方，
            结果终点坐标为 (100, 100)。
        """
        angle_rad = math.radians(angle)  # 将角度转换为弧度
        x = int(center[0] + radius * math.cos(angle_rad))
        y = int(center[1] + radius * math.sin(angle_rad))
        return (x, y)

def run_scrcpy():
    # 替换为你的设备ID
    device_id = "emulator-5554"
    max_width = 1080
    max_fps = 60
    bit_rate = 2000000000

    client = scrcpy.Client(device=device_id, max_width=max_width, max_fps=max_fps, bitrate=bit_rate)
    client.start(threaded=True)

    # 获取当前工作目录
    current_directory = os.getcwd()

    # 命令参数
    command = f'start cmd.exe /k {current_directory}/tools/scrcpy.exe -s {device_id} -m 1080 --window-title wzry_ai'

    print("执行命令：", command)

    # 执行命令
    subprocess.run(command, shell=True)

    return client


# 使用示例
if __name__ == "__main__":

    controller = AndroidController(run_scrcpy())

    try:
        # 模拟同时执行点击、滑动和长按操作
        # actions = [
        #     {'action': 'action1', 'type': 'down', 'timeout': 10},
        #     {'action': 'action1', 'type': 'down', 'timeout': 3},  # 测试重复点击同一位置
        #     {'action': 'move', 'type': 'swipe', 'angle': 90},
        #     {'action': 'action3', 'type': 'long_press', 'duration': 1500}
        # ]
        # controller.execute_actions(actions)
        # time.sleep(15)  # 保持程序运行以观察效果
        while(True):
            random_angle = random.randint(0, 359)
            actions = [
                {'action': 'move', 'type': 'swipe', 'angle': random_angle}
            ]
            controller.execute_actions(actions)
            # time.sleep(1)  # 保持程序运行以观察效果
    finally:
        controller.stop()

        # scrcpy.exe -s 192.168.10.4:5555 --turn-screen-off --stay-awake -m 768 --window-title xxx --always-on-top
        # scrcpy.exe -s 528e7355 -m 1080 --window-title wzry_ai
