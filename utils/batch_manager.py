class BatchManager:
    def __init__(self):
        try:
            with open("batch.txt", "r") as f:
                self.counter = int(f.read())
        except:
            self.counter = 1

    def new_batch(self):
        batch_id = f"B{self.counter:03d}"
        self.counter += 1

        with open("batch.txt", "w") as f:
            f.write(str(self.counter))

        return batch_id