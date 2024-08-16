import random
import time
import json
import os

from argparses import globalInfo


def train_dqn():
    for episode in range(1000):
        loss = random.uniform(0, 1)
        reward = random.uniform(0, 10)

        loss_data = {
            'title': 'DQN Loss',
            'description': 'Loss over time during DQN training',
            'x_label': 'Episodes',
            'y_label': 'Loss',
            'x_data': [episode],
            'y_data': [loss]
        }

        reward_data = {
            'title': 'DQN Reward',
            'description': 'Reward over time during DQN training',
            'x_label': 'Episodes',
            'y_label': 'Reward',
            'x_data': [episode],
            'y_data': [reward]
        }

        globalInfo.update_data_file([loss_data, reward_data])

        time.sleep(1)  # 模拟训练时间
        print("aaaa")


if __name__ == '__main__':
    train_dqn()
