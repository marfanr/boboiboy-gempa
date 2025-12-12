from typing import Literal

import numpy as np

from transport.win import WinTransportServer
import threading
import obspy
import time
import pywintypes

shutdown_event = threading.Event()


class Master:
    def __init__(self, parallel_count: int = 10):
        self.parallel_count = parallel_count
        self.threads = []
        self.clients = []

    def handle_node(self, node_id):
        try:
            transport = WinTransportServer(node_id)
            transport.start()
            print(f"[Master] Worker {node_id} connected.")

            while True:
                try:
                    data = transport.recv()
                    print("Received:", data)
                    mode = data[0]
                    length = int.from_bytes(data[1:5], "big")
                    match mode:
                        case 1:
                            client_job = "inference"
                            if data[5] == 0x33:
                                client_job = "collector"
                            print(f"[Master] received greeting from {node_id} as a {client_job}")
                            self.clients.append((client_job, client_job))

                        case 2:
                            print(f"[Master] collector {node_id} send data")
                    # size = data[1]
                    # payload = data[5:]
                    # print(
                    #     f"[Master] Worker {node_id} to {target} size {length} received {np.frombuffer(payload, dtype=np.float32)}")

                except pywintypes.error as e:
                    # error 109 = client disconnect
                    if e.winerror == 109:
                        print(f"[Master] Worker {node_id} disconnected (pipe ended).")
                    else:
                        print(f"[Master] Worker {node_id} pipe error: {e}")
                    break

        except Exception as e:
            print(f"[Master] Worker {node_id} died: {e}")

    def start_worker(self, node_id):
        t = threading.Thread(target=self.handle_node, args=(node_id,))
        t.daemon = True
        t.start()
        return t

    def run(self):
        print("Starting Master...")

        # initial spawn
        for i in range(self.parallel_count):
            self.threads.append(self.start_worker(i))

        try:
            while not shutdown_event.is_set():

                # refill workers
                alive = []
                for idx, t in enumerate(self.threads):
                    if t.is_alive():
                        alive.append(t)
                    else:
                        print(f"Worker {idx} is dead. Restarting...")
                        new_thread = self.start_worker(idx)
                        alive.append(new_thread)

                self.threads = alive

                time.sleep(0.8)

        except KeyboardInterrupt:
            print("KeyboardInterrupt caught.")
            shutdown_event.set()

        print("Master exiting...")
