import datetime
import threading

from memory import ReplayMemory


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class GlobalInfo:
    def __init__(self, batch_size=64, buffer_capacity=10000):
        self.batch_size = batch_size
        self._info = {}
        self.ppo_memory = ReplayMemory(buffer_capacity)
        self.td3_memory = ReplayMemory(buffer_capacity)
        self.dqn_memory = ReplayMemory(buffer_capacity)
        self.lock = threading.Lock()

    def set_value(self, key, value):
        self._info[key] = value

    def get_value(self, key):
        return self._info.get(key, None)

    # -------------------------------对局状态-------------------------------------
    def set_game_start(self):
        self.set_value('start_game', True)

    def is_start_game(self):
        start_game = self.get_value('start_game')
        if start_game is None:
            return False
        else:
            return start_game

    def set_game_end(self):
        self.set_value('start_game', False)

    # -------------------------------ppo经验池-------------------------------------
    def store_transition_ppo(self, *args):
        self.lock.acquire()
        try:
            self.ppo_memory.push(*args)
        finally:
            self.lock.release()

    def is_memory_bigger_batch_size_ppo(self):
        self.lock.acquire()
        try:
            if len(self.ppo_memory) < self.batch_size:
                return False
            else:
                return True
        finally:
            self.lock.release()

    def random_batch_size_memory_ppo(self):
        self.lock.acquire()
        try:
            transitions = self.ppo_memory.sample(self.batch_size)
            return transitions
        finally:
            self.lock.release()

    # -------------------------------td3经验池-------------------------------------
    def store_transition_td3(self, *args):
        self.td3_memory.push(*args)

    def is_memory_bigger_batch_size_td3(self):
        if len(self.td3_memory) < self.batch_size:
            return False
        else:
            return True

    def random_batch_size_memory_td3(self):
        transitions = self.td3_memory.sample(self.batch_size)
        return transitions

    # -------------------------------dqn经验池-------------------------------------
    def store_transition_dqn(self, *args):
        self.dqn_memory.push(*args)

    def is_memory_bigger_batch_size_dqn(self):
        if len(self.dqn_memory) < self.batch_size:
            return False
        else:
            return True

    def random_batch_size_memory_dqn(self):
        transitions = self.dqn_memory.sample(self.batch_size)
        return transitions
