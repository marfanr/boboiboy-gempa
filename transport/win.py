import win32pipe


class WinTransportProducer:
    def __init__(self, name: str):
        self.process_name = name
        self.pipe_name = r"\\.\pipe\{}".format(self.process_name)
        self.pipe = win32pipe.CreateNamedPipe(self.pipe_name,
                                              win32pipe.PIPE_ACCESS_DUPLEX,
                                              win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                                              1, 65536, 65536, 0, None
                                              )

    def wait(self):
        print("Waiting for client...")
        win32pipe.ConnectNamedPipe(self.pipe, None)
        print("Client connected.")

class WinSharedMemoryManager:
    def __init__(self, name: str, pipe: WinTransportProducer|None = None, size: int = 1024*1024):
        self.process_name = name