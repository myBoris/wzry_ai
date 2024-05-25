import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
import win32gui
from PyQt5.QtWidgets import QApplication
import torchvision.transforms as transforms
import torch.nn.functional as F

left_actions = {
    '0': {'action_id': 'none', 'action_name': '无操作', 'type': 'none'},
    '1': {'action_id': 'move', 'action_name': '移动', 'type': 'swipe'}

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
    '18': {'action_id': 'skill_equipment', 'action_name': '装备技能'},
    '19': {'action_id': 'buy_equipment_1', 'action_name': '购买装备1'},
    '20': {'action_id': 'buy_equipment_2', 'action_name': '购买装备2'}
}

skill_actions = {
    '0': {'action_id': 'click', 'action_name': '点击'},
    '1': {'action_id': 'swipe', 'action_name': '滑动'},
    '2': {'action_id': 'long_press', 'action_name': '长按'}
}

def conver_model_result_to_action(action):
    action1_logits, angle1_logits, action2_logits, type2_logits, angle2_logits, duration2_logits = split_actions(action)

    # 左手的操作
    # 获取最可能的action和type
    action1 = torch.argmax(action1_logits, dim=1)  # 得到0-2之间的整数
    angle1 = torch.argmax(angle1_logits, dim=1)  # 得到0-359之间的整数

    # 右手的操作
    # 获取最可能的action和type
    action2 = torch.argmax(action2_logits, dim=1)  # 得到0-20之间的整数
    type2 = torch.argmax(type2_logits, dim=1)  # 得到0-2之间的整数
    angle2 = torch.argmax(angle2_logits, dim=1)  # 得到0-359之间的整数
    duration2 = F.relu(duration2_logits)

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

    # print("action", actions)

    return actions


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param_count = parameter.numel()
        print(f"Layer {name}: {param_count} parameters")
        total_params += param_count
    print(f"Total parameters: {total_params}")
    return total_params

def combine_actions(action1, angle1, action2, type2, angle2, duration2):
    # 将多个动作组合成一个向量
    return torch.cat((action1, angle1, action2, type2, angle2, duration2), dim=-1)

def split_actions(combined_action):
    # 将组合的动作向量分解为单独的动作

    # 左手的操作，有4个，(无操作,移动，购买装备1，购买装备2)
    action1 = combined_action[:, :2]
    # 360度
    angle1 = combined_action[:, 2:362]
    # 右手的操作，有21个，
    # (无操作,回城，恢复，召唤师技能，攻击，   5个
    # 攻击小兵，攻击塔，发起进攻，开始撤退，请求集合， 5个
    # 1技能，2技能，3技能，4技能，升级1技能， 5个
    # 升级2技能，升级3技能， 升级4技能, 装备技能) 4个
    # 购买装备1，购买装备2) 2个
    action2 = combined_action[:, 362:383]
    # 技能有(点击，滑动，长按)三种释放方式
    type2 = combined_action[:, 383:386]
    # 360度
    angle2 = combined_action[:, 386:746]
    # 长按时间
    duration2 = combined_action[:, 746:]
    return action1, angle1, action2, type2, angle2, duration2

# 使用示例
if __name__ == "__main__":
    pass