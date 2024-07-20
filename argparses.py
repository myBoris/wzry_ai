# config.py
import argparse

import torch

from globalInfo import GlobalInfo

# 移动坐标和滑动半径
move_actions_detail = {
    1: {'action_name': '移动', 'position': (0.164, 0.798), 'radius': 200}
}

# 点击坐标
#  购买装备1， 购买装备2，发起进攻，开始撤退，请求集合，升级1技能，升级2技能，升级3技能，升级4技能
info_actions_detail = {
    1: {'action_name': '购买装备1', 'position': (0.133, 0.4), 'radius': 0},
    2: {'action_name': '购买装备2', 'position': (0.133, 0.51), 'radius': 0},
    3: {'action_name': '发起进攻', 'position': (0.926, 0.14), 'radius': 0},
    4: {'action_name': '开始撤退', 'position': (0.926, 0.22), 'radius': 0},
    5: {'action_name': '请求集合', 'position': (0.926, 0.31), 'radius': 0},
    6: {'action_name': '升级1技能', 'position': (0.668, 0.772), 'radius': 0},
    7: {'action_name': '升级2技能', 'position': (0.717, 0.59), 'radius': 0},
    8: {'action_name': '升级3技能', 'position': (0.8, 0.48), 'radius': 0}
}

# 无操作, 攻击，攻击小兵，攻击塔，回城，恢复，装备技能, 1技能，2技能，3技能,
attack_actions_detail = {
    1: {'action_name': '攻击', 'position': (0.85, 0.85), 'radius': 0},
    2: {'action_name': '攻击小兵', 'position': (0.776, 0.91), 'radius': 0},
    3: {'action_name': '攻击塔', 'position': (0.88, 0.71), 'radius': 0},
    4: {'action_name': '回城', 'position': (0.518, 0.9), 'radius': 0},
    5: {'action_name': '恢复', 'position': (0.579, 0.9), 'radius': 0},
    6: {'action_name': '装备技能', 'position': (0.84, 0.39), 'radius': 0},
    7: {'action_name': '召唤师技能', 'position': (0.64, 0.9), 'radius': 50},
    8: {'action_name': '1技能', 'position': (0.71, 0.874), 'radius': 100},
    9: {'action_name': '2技能', 'position': (0.76, 0.69), 'radius': 100},
    10: {'action_name': '3技能', 'position': (0.844, 0.58), 'radius': 100}
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iphone_id', type=str, default='528e7355', help="device_id")
    parser.add_argument('--window_title', type=str, default='wzry_ai', help="device_id")
    parser.add_argument('--device_id', type=str, default='cuda:0', help="device_id")
    parser.add_argument('--memory_size', type=int, default=10000, help="Replay memory size")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--epsilon', type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help="Exploration rate decay")
    parser.add_argument('--epsilon_min', type=float, default=0.01, help="Minimum exploration rate")
    parser.add_argument('--model_path', type=str, default=None, help="Path to the model to load")
    parser.add_argument('--num_episodes', type=int, default=100, help="Number of episodes to collect data")
    parser.add_argument('--target_update', type=int, default=10, help="Number of episodes to collect data")

    return parser.parse_args()


# 解析参数并存储在全局变量中
args = get_args()

device = torch.device(args.device_id if torch.cuda.is_available() else 'cpu')

# 全局状态
globalInfo = GlobalInfo(batch_size=args.batch_size, buffer_capacity=args.memory_size)
