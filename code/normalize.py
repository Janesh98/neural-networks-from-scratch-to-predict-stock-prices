import numpy as np

class Normalize:
    """ normalizes the data between 0 and 1
        and reverts data back to original values """
    def __init__(self, data, max=False, minmax=False):
        self.minmax = minmax
        if max:
            self.factor = self.get_max(data)
        if self.minmax:
            self.minim = min(data)
            self.factor = self.get_minmax(data)
        else:
            self.factor = self.get_powers(data)

    def get_minmax(self, data):
        return max(data) - min(data)

    def get_max(self, data):
        return max(data)

    def get_powers(self, data):
        return 10 ** len(str(round(max(data))))

    # scale data between 0 and 1
    def normalize_data(self, data):
        data = np.array(data, dtype=float)
        if self.minmax:
            return self.normalize_minmax(data)
        else:
            return np.divide(data, self.factor)

    # revert data back to original values
    def denormalize_data(self, data):
        if self.minmax:
            return self.denormalize_minmax(data)
        else:
            return np.dot(data, self.factor)

    # minmax normalization
    def normalize_minmax(self, data):
        data = np.array(data, dtype=float)
        for i in data:
            i -= self.minim
        return np.divide(data, self.factor)

    # minmax denormalization
    def denormalize_minmax(self, data):
        y = np.dot(data, self.factor)
        print(self.minim)
        for i in y:
            i += self.minim
        return y