from collections import namedtuple
import random

# Transition - 一个命名元组(named tuple)用于表示环境中的单次状态迁移(single transition)。
#     该类的作用本质上是将状态-动作对[(state, action) pairs]映射到他们的下一个结果，即[(next_state, action) pairs]，
# 其中的 状态(state)是指从屏幕上获得的差分图像块(screen diifference image)。
#
# ReplayMemory - 一个大小有限的循环缓冲区，用于保存最近观察到的迁移(transition)。
# 该类还实现了一个采样方法 .sample() 用来在训练过程中随机的选择一个迁移批次(batch of transitions)。

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
