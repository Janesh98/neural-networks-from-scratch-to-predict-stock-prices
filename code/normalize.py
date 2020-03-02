import numpy as np

class Normalize:
    """ normalizes the data between 0 and 1
        and reverts data back to original values """
    def __init__(self, data):
        self.factor = self.normalize_factor(data)

    # finds a factor one bigger than the length of the data
    # e.g.
    # if max value in data is 100, factor is 1000
    # if max value in data is 1500, factor is 10000
    def normalize_factor(self, data):
        return max(data)

    # scale data between 0 and 1
    def normalize_data(self, data):
        data = np.array(data, dtype=float)
        return data / self.factor

    # revert data back to original values
    def denormalize_data(self, data):
        return data * self.factor