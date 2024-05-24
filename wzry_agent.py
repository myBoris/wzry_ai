import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from methodutil import load_and_preprocess_image
from model import WzryNet


class Agent:
    def __init__(self, action_size=747, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, lr=0.001,
                 batch_size=4, memory_size=10000, target_update=10, load_model=False):

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = WzryNet().to(self.device)
        self.target_model = WzryNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if load_model:
            self.load("../src/wzry_ai.pt")

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # 返回随机动作
            return torch.randn(1, self.action_size).to(self.device)
        state = load_and_preprocess_image(state).to(self.device)
        with torch.no_grad():
            action = self.model(state)
        return action

    def replay(self, epoch):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = load_and_preprocess_image(state).to(self.device)
            next_state = load_and_preprocess_image(next_state).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            done = torch.FloatTensor([done]).to(self.device)

            target = reward
            if not done.item():
                with torch.no_grad():
                    target_action = self.target_model(next_state)
                    target = reward + self.gamma * torch.max(target_action)

            predicted_action = self.model(state)

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
            self.save(f"src/model_episode_{epoch}.pt")

    def load(self, name):
        if os.path.exists(name):
            self.model.load_state_dict(torch.load(name, map_location=self.device), strict=False)
            print(f"Loaded model from {name}")
        else:
            print(f"No model found at {name}. Starting training from scratch.")

    def save(self, name):
        torch.save(self.model.state_dict(), name)
        print(f"Model saved to {name}")
