class BatchManager:
    def __init__(self):
        self.counter = 1

    def new_batch(self):
        batch_id = f"B{self.counter:03d}"
        self.counter += 1
        return batch_id