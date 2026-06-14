import threading
from queue import Queue, Empty
from database.db import save_to_influx

class InfluxWorker:
    def __init__(self):
        self.queue = Queue(maxsize=1000)
        self.running = True
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:

            try:
                result = self.queue.get(timeout=1)
            except Empty:
                continue

            if result is None:
                break

            try:
                save_to_influx(result)
            except Exception as e:
                print("[Influx ERROR]", e)

            self.queue.task_done()

    def write(self, data):

        try:
            self.queue.put(data, block=False)
        except:
            # drop oldest behavior can be added here if needed
            print("[Influx WARN] queue full, dropping data")

    def stop(self):
        self.running = False
        self.queue.put(None)

        if self.thread:
            self.thread.join(timeout=2)