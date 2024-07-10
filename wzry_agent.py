import os
import random
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from model import WzryNet


class Agent:
    def __init__(self, action_size=747, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, lr=0.001,
                 batch_size=4, memory_size=10000, target_update=10, train_model_path="src/wzry_ai.pt"):

        torch.backends.cudnn.enabled = False

        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.target_update = target_update
        self.target_update_count = 0
        self.train_model_path = train_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = WzryNet().to(self.device)
        self.target_model = WzryNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if os.path.exists(train_model_path):
            self.load(train_model_path)

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # 返回随机动作
            return torch.randn(1, self.action_size).to(self.device)
        with torch.no_grad():
            tmp_state_640_640 = self.preprocess_image(state)
            action = self.model(tmp_state_640_640)
        return action

    def preprocess_image(self, image, target_size=(640, 640)):
        # 调整图像大小
        resized_image = cv2.resize(image, target_size)
        # 转换为张量并调整维度顺序 [height, width, channels] -> [channels, height, width]
        tensor_image = torch.from_numpy(resized_image).float().permute(2, 0, 1)
        return tensor_image.to(self.device).unsqueeze(0)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            reward = torch.FloatTensor([reward]).to(self.device)
            done = torch.FloatTensor([done]).to(self.device)

            target = reward
            if not done.item():
                with torch.no_grad():
                    tmp_next_state_640_640 = self.preprocess_image(next_state)
                    target_action = self.target_model(tmp_next_state_640_640)
                    target = reward + self.gamma * torch.max(target_action)

            tmp_state_640_640 = self.preprocess_image(state)
            predicted_action = self.model(tmp_state_640_640)

            if target.dim() == 1:
                target = target.expand_as(predicted_action)

            loss = F.mse_loss(predicted_action, target)
            dqn_loss = torch.mean(loss)

            self.optimizer.zero_grad()
            dqn_loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.target_update_count += 1
        if self.target_update_count == self.target_update:
            self.target_update_count = 0
            self.update_target_model()
            self.save(self.train_model_path)

    def load(self, name):
        if os.path.exists(name):
            self.model.load_state_dict(torch.load(name, map_location=self.device), strict=True)
            print(f"Loaded model from {name}")
        else:
            print(f"No model found at {name}. Starting training from scratch.")

    def save(self, name):
        torch.save(self.model.state_dict(), name)
        print(f"Model saved to {name}")
