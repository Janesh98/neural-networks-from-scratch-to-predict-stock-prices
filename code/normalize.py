import numpy as np

class Normalize:
    """ normalizes the data between 0 and 1
        and reverts data back to original values """
    def __init__(self, data):
        self.factor = self.normalize_factor(data)

    def normalize_factor(self, data):
        return max(data)

    # scale data between 0 and 1
    def normalize_data(self, data):
        data = np.array(data, dtype=float)
        return np.divide(data, self.factor)

    # revert data back to original values
    def denormalize_data(self, data):
        return np.dot(data, self.factor)