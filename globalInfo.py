import datetime
import threading


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

    # -------------------------------对局时间-------------------------------------
    def set_start_game_time(self):
        self.set_value('game_time', datetime.datetime.now())

    def get_game_time_pass(self):
        game_time = self.get_value('back_home')
        if game_time is None:
            return 0

        current_time = datetime.datetime.now()
        elapsed_time = (current_time - game_time).total_seconds()
        return elapsed_time

    # -------------------------------回城状态-------------------------------------
    def set_back_home_time(self):
        self.set_value('back_home', datetime.datetime.now())

    def is_back_home(self):
        back_home = self.get_value('back_home')
        if back_home is None:
            return False
        else:
            return True

    def is_back_home_over(self):
        back_home = self.get_value('back_home')
        if back_home is None:
            return True

        current_time = datetime.datetime.now()
        elapsed_time = (current_time - back_home).total_seconds()

        if elapsed_time > 8:
            self.set_value('back_home', None)
            return True
        else:
            return False

    # -------------------------------state-------------------------------------
    def set_global_frame(self, globalFrame):
        self.lock.acquire()
        try:
            self.set_value("globalFrame", globalFrame)
        finally:
            self.lock.release()

    def get_global_frame(self):
        self.lock.acquire()
        try:
            globalFrame = self.get_value("globalFrame")
            return globalFrame
        finally:
            self.lock.release()
