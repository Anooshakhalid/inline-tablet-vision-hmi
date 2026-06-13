import os

class BatchManager:
    def __init__(self):
        self.file = os.path.join(os.path.dirname(__file__), "batch.txt")

        try:
            with open(self.file, "r") as f:
                content = f.read().strip()
                self.counter = int(content) if content else 1
        except:
            self.counter = 1

    def new_batch(self):
        batch_id = f"B{self.counter:03d}"
        self.counter += 1

        with open(self.file, "w") as f:
            f.write(str(self.counter))

        return batch_id