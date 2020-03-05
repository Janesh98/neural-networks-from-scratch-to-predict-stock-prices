import numpy as np

class Normalize:
    """ normalizes the data between 0 and 1
        and reverts data back to original values """
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

"""class Normalize:
    #normalizes the data between 0 and 1
    #    and reverts data back to original values
    def __init__(self, data):
        self.factor = self.normalize_factor(data)
        self.minim = min(data)

    def normalize_factor(self, data):
        return (max(data) - min(data))

    # scale data between 0 and 1
    def normalize_data(self, data):
        data = np.array(data, dtype=float)
        for i in data:
            i -= self.minim
        return np.divide(data, self.factor)

    # revert data back to original values
    def denormalize_data(self, data):
        y = np.dot(data, self.factor)
        print(self.minim)
        for i in y:
            i += self.minim
        return y"""