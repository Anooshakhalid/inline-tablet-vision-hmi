import threading
from queue import Queue
from database.db import save_to_influx

class InfluxWorker:
    def __init__(self):
        self.queue = Queue()
        self.running = True

    def start(self):
        thread = threading.Thread(target=self._worker, daemon=True)
        thread.start()

    def _worker(self):
        while self.running:
            result = self.queue.get()

            if result is None:
                break

            try:
                save_to_influx(result)
            except Exception as e:
                print("[Influx ERROR]", e)

            self.queue.task_done()

    def write(self, data):
        self.queue.put(data)

    def stop(self):
        self.queue.put(None)
        self.running = False