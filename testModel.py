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
    image = cv2.resize(image, model_input_size, interpolation=cv2.INTER_LINEAR)

    # 将数据类型转换为float32并归一化像素值
    image = image.astype('float32') / 255.0

    # 转换为torch张量，并添加一个batch维度
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)  # 增加batch维度，因为模型期望批处理输入

    return image

def run_scrcpy():
    device_id = "528e7355"
    max_width = 1080
    max_fps = 60
    bit_rate = 2000000000

    client = scrcpy.Client(device=device_id, max_width=max_width, max_fps=max_fps, bitrate=bit_rate)
    client.start(threaded=True)

    return client

def conver_model_result_to_action(output1, output2):
    left_actions = {
        '0': {'action_id': 'none', 'action_name': '无操作', 'type': 'none'},
        '1': {'action_id': 'move', 'action_name': '移动', 'type': 'swipe'},
        '2': {'action_id': 'buy_equipment_1', 'action_name': '购买装备1', 'type': 'click'},
        '3': {'action_id': 'buy_equipment_2', 'action_name': '购买装备2', 'type': 'click'}
    }

    right_actions = {
        '0': {'action_id': 'none', 'action_name': '无操作'},
        '1': {'action_id': 'back_base', 'action_name': '回城'},
        '2': {'action_id': 'restore_health', 'action_name': '恢复'},
        '3': {'action_id': 'skill', 'action_name': '召唤师技能'},
        '4': {'action_id': 'attack', 'action_name': '攻击'},
        '5': {'action_id': 'attack_pawn', 'action_name': '攻击小兵'},
        '6': {'action_id': 'attack_tower', 'action_name': '攻击塔'},
        '7': {'action_id': 'attention_1', 'action_name': '发起进攻'},
        '8': {'action_id': 'attention_2', 'action_name': '开始撤退'},
        '9': {'action_id': 'attention_3', 'action_name': '请求集合'},
        '10': {'action_id': 'skill_1', 'action_name': '1技能'},
        '11': {'action_id': 'skill_2', 'action_name': '2技能'},
        '12': {'action_id': 'skill_3', 'action_name': '3技能'},
        '13': {'action_id': 'skill_4', 'action_name': '4技能'},
        '14': {'action_id': 'add_skill_1', 'action_name': '升级1技能'},
        '15': {'action_id': 'add_skill_2', 'action_name': '升级2技能'},
        '16': {'action_id': 'add_skill_3', 'action_name': '升级3技能'},
        '17': {'action_id': 'add_skill_4', 'action_name': '升级4技能'},
        '18': {'action_id': 'skill_equipment', 'action_name': '装备技能'}
    }

    skill_actions = {
        '0': {'action_id': 'click', 'action_name': '点击'},
        '1': {'action_id': 'swipe', 'action_name': '滑动'},
        '2': {'action_id': 'long_press', 'action_name': '长按'}
    }

    # 解包输出
    action1_logits, angle1_logits = output1

    # 左手的操作
    # 获取最可能的action和type
    action1 = torch.argmax(action1_logits, dim=1)  # 得到0-4之间的整数
    angle1 = torch.argmax(angle1_logits, dim=1)  # 得到0-359之间的整数

    # 右手的操作
    action2_logits, type2_logits, angle2_logits, duration2 = output2

    # 获取最可能的action和type
    action2 = torch.argmax(action2_logits, dim=1)  # 得到0-18之间的整数
    type2 = torch.argmax(type2_logits, dim=1)  # 得到0-3之间的整数
    angle2 = torch.argmax(angle2_logits, dim=1)  # 得到0-359之间的整数

    actions = [
        {
            'action': left_actions[str(action1.item())]['action_id'],
            'type': left_actions[str(action1.item())]['type'],
            'angle': int(angle1)
        },
        {
            'action': right_actions[str(action2.item())]['action_id'],
            'type': skill_actions[str(type2.item())]['action_id'],
            'angle': int(angle2),
            'duration': int(duration2)
        }
    ]

    return actions



# 使用示例
if __name__ == "__main__":
    # 示例使用
    controller = AndroidController(run_scrcpy())

    try:
        # 模型
        # model = torch.load('src/wzry_ai.pt')

        # model = WzryNet()
        # model.eval()  # 设置为评估模式

        while(True):
            # 记录开始时间
            start_time = time.time()

            # 测试，不应该放这里，要放外面
            model = WzryNet()
            model.eval()  # 设置为评估模式

            image = methodutil.screenshot("wzry_ai")
            processed_image = load_and_preprocess_image(image, model_input_size=(640, 640))

            # 记录并打印这一步骤的时间
            step1_time = time.time()
            print(f"第一步运行时间: {step1_time - start_time:.3f} 秒")

            with torch.no_grad():
                output1, output2 = model(processed_image)
                # 记录并打印这一步骤的时间
                step2_time = time.time()
                print(f"第二步运行时间: {step2_time - step1_time:.3f} 秒")

                actions = conver_model_result_to_action(output1, output2)
                # 记录并打印这一步骤的时间
                step3_time = time.time()
                print(f"第三步运行时间: {step3_time - step2_time:.3f} 秒")
                print(actions)

            controller.execute_actions(actions)
            # time.sleep(1)  # 保持程序运行以观察效果
            # 打印总运行时间
            total_time = time.time() - start_time
            print(f"总运行时间: {total_time:.3f} 秒")
    finally:
        pass
