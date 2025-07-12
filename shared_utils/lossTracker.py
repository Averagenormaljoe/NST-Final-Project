class LossTracker:
    def __init__(self):
        self.losses = {}

    def add(self, key, loss):
        self.losses.setdefault(key, []).append(loss)
    def get_loss(self, key):
        return self.losses.get(key, [])

