import numpy as np

class History:
    def __init__(self, config):
        self.history = np.zeros([config.history_length, config.input_size], dtype=np.int32)

    def add(self, state):
        self.history[:-1] = self.history[1:]
        self.history[-1] = state

    def reset(self):
        self.history *= 0

    def get(self):
        return self.history
