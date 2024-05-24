import datetime


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class GlobalInfo:
    def __init__(self):
        self._info = {}

    def set_value(self, key, value):
        self._info[key] = value

    def get_value(self, key):
        return self._info.get(key, None)

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

    def set_start_time(self, key):
        self.set_value(key, datetime.datetime.now())

    def has_time_elapsed(self, key, seconds):
        start_time = self.get_value(key)
        if start_time is None:
            raise ValueError(f"Start time for {key} has not been initialized.")

        current_time = datetime.datetime.now()
        elapsed_time = (current_time - start_time).total_seconds()
        return elapsed_time > seconds


# 使用示例
if __name__ == "__main__":
    g1 = GlobalInfo()

    # 初始化时间
    g1.set_start_time('start_time_key')

    # 等待几秒钟后检查是否超过指定的秒数
    import time

    time.sleep(3)  # 等待3秒

    print(g1.has_time_elapsed('start_time_key', 2))  # 输出: True, 因为等待了3秒，超过了2秒
    print(g1.has_time_elapsed('start_time_key', 5))  # 输出: False, 因为等待了3秒，未超过5秒

    # 验证单例
    g2 = GlobalInfo()
    print(g1 is g2)  # 输出: True，证明g1和g2是同一个实例
