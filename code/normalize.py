import numpy as np

class Normalize:
    """normalizes the data between 0 and 1
    and reverts data back to original values"""
    def __init__(self, data, max=False):
        if max:
            self.factor = self.normalize_max(data)
        else:
            self.factor = self.normalize_powers(data)

    def normalize_max(self, data):
        return max(data)

    def normalize_powers(self, data):
        return 10 ** len(str(round(max(data))))

    # scale data between 0 and 1
    def normalize_data(self, data):
        data = np.array(data, dtype=float)
        return np.divide(data, self.factor)

    # revert data back to original values
    def denormalize_data(self, data):
        return np.dot(data, self.factor)

class MinMax:
    """normalizes the data between 0 and 1
    and reverts data back to original values"""
    def __init__(self, data):
        self.minim = min(data)
        self.factor = self.get_minmax(data)

    def get_minmax(self, data):
        return max(data) - min(data)

    # scale data between 0 and 1
    def normalize_data(self, data):
        data = np.array(data, dtype=float)
        return self.normalize_minmax(data)

    # revert data back to original values
    def denormalize_data(self, data):
        return self.denormalize_minmax(data)

    # minmax normalization
    def normalize_minmax(self, data):
        data = np.array(data, dtype=float)
        for i in data:
            i -= self.minim
        return np.divide(data, self.factor)

    # minmax denormalization
    def denormalize_minmax(self, data):
        y = np.dot(data, self.factor)
        for i in y:
            i += self.minim
        return y