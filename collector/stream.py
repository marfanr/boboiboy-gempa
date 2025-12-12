import obspy
from transport.win import WinTransportProducer
import numpy as np

class ObspyStreamWorker:
    def __init__(self, client_id: int, station: str):
        self.station = station
        self.id = int(client_id)
        self.producer = WinTransportProducer(name=str(client_id))
        print(f"Station name: {station}")

    def send_packet(self, op: int, payload: bytes):
        length = len(payload)
        packet = bytes([op]) + length.to_bytes(4, "big") + payload
        self.producer.send(packet)

    def send_greeting(self):
        """
        Send a greeting to notify server this client job is a data collector
        :return:
        """
        self.send_packet(1, bytes([0x33]))

    def run(self):
        try:
            self.producer.wait()
            self.send_greeting()
            # self.send_packet(2, np.array([2,3], dtype=np.float32).tobytes())
            while True:
                pass

        except KeyboardInterrupt:
            print("Shutdown requested")