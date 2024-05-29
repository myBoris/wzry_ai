import threading
import keyboard

from globalInfo import GlobalInfo

class KeyboardListener:
    def __init__(self):
        self._stop_event = threading.Event()
        self._listener_thread = threading.Thread(target=self._listen_to_keyboard)
        # 全局状态
        self.globalInfo = GlobalInfo()
        self._lock = threading.Lock()

    def _listen_to_keyboard(self):
        while not self._stop_event.is_set():
            event = keyboard.read_event()
            if event.event_type == keyboard.KEY_DOWN:
                if event.name == 'esc':
                    with self._lock:
                        recordImgFlg = self.globalInfo.get_value("recordImg")
                        if recordImgFlg is None:
                            recordImgFlg = False
                            self.globalInfo.set_value("recordImg", recordImgFlg)
                        else:
                            recordImgFlg = not recordImgFlg
                            self.globalInfo.set_value("recordImg", recordImgFlg)
                elif event.name == '1':
                    with self._lock:
                        self.globalInfo.set_value("reword", 1)
                elif event.name == '2':
                    with self._lock:
                        self.globalInfo.set_value("reword", 2)
                elif event.name == '3':
                    with self._lock:
                        self.globalInfo.set_value("reword", 3)
                elif event.name == '4':
                    with self._lock:
                        self.globalInfo.set_value("reword", -1)
                elif event.name == '5':
                    with self._lock:
                        self.globalInfo.set_value("reword", -2)
                elif event.name == '6':
                    with self._lock:
                        self.globalInfo.set_value("reword", -3)

    def start(self):
        self._listener_thread.start()

    def stop(self):
        self._stop_event.set()
        self._listener_thread.join()


# 示例使用方法
if __name__ == "__main__":
    listener = KeyboardListener()
    listener.start()

    try:
        while True:
            pass  # 主线程可以执行其他任务
    except KeyboardInterrupt:
        listener.stop()
        print("Keyboard listener stopped.")
