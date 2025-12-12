import win32pipe
import win32file
import pywintypes
import time

class WinTransportServer:
    def __init__(self, name):
        self.pipe_name = r"\\.\pipe\GempaPipe{}".format(name)
        self.handle = None

    def start(self):
        print("[Master] Creating pipe:", self.pipe_name)
        self.handle = win32pipe.CreateNamedPipe(
            self.pipe_name,
            win32pipe.PIPE_ACCESS_DUPLEX,
            win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
            1, 65536, 65536,
            0, None
        )

        win32pipe.ConnectNamedPipe(self.handle, None)

    def send(self, data: bytes):
        win32file.WriteFile(self.handle, data)

    def recv(self):
        result, data = win32file.ReadFile(self.handle, 65536)
        return data

    def close(self):
        self.handle = None

    def isOpen(self):
        return self.handle is not None


class WinTransportProducer:
    def __init__(self, name: str):
        self.pipe_name = r"\\.\pipe\GempaPipe{}".format(name)
        self.handle = None

    def wait(self):
        print("[Node] Waiting server...")

        while True:
            try:
                self.handle = win32file.CreateFile(
                    self.pipe_name,
                    win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                    0, None,
                    win32file.OPEN_EXISTING,
                    0, None
                )
                break
            except pywintypes.error:
                time.sleep(0.1)  # retry sampai server siap

        print("[Node] Connected to server")

    def send(self, data: bytes):
        win32file.WriteFile(self.handle, data)

    def recv(self):
        result, data = win32file.ReadFile(self.handle, 65536)
        return data

#
# class WinSharedMemoryManager:
#     def __init__(self, name: str, pipe: WinTransportProducer | None = None, size: int = 1024 * 1024):
#         self.process_name = name
