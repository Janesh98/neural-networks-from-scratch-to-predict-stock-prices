import numpy as np

def mape(actual, prediction): 
    # mean absolute percentage error (MAPE)
    return np.mean(np.abs((actual - prediction) / actual)) * 100

def mse(actual, prediction):
    mse = np.mean((actual - prediction)**2)
    return mse

def rmse(actual, prediction):
    return np.sqrt(((actual - prediction) ** 2).mean())

def train_test_split(df, split=0.75):
    # if split=0.75, splits data into 75% training, 25% test
    # provides targets for training and accuracy measurments
    max_index = round((len(df) - 1) * split)

    # adjusted close price [2 days ago, 1 day ago]
    train_inputs = [[df[i-2], df[i-1]] for i in range(2, max_index)]
    # target is the next day for a given input above
    # e.g inputs = [day1, day2], [day2, day3]
    #     targets = [day3, day4]
    train_targets = [i for i in df[2 : max_index]]

    assert len(train_inputs) == len(train_targets)

    test_inputs = [[df[i-2], df[i-1]] for i in range(max_index + 2, len(df))]
    test_targets = [i for i in df[max_index + 2:]]

    assert len(test_inputs) == len(test_targets)

    return np.array(train_inputs), np.array(train_targets), np.array(test_inputs), np.array(test_targets)

def to_3d(data):
    # reshape data to [inputs, timesteps, features]
    train = data.reshape(data.shape[0], data.shape[1], 1)
    return train