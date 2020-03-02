import numpy as np
import matplotlib.pyplot as plt
from stock import get_stock_data
from normalize import Normalize
from NeuralNetwork import mse, rmse, mape

class RNN():
    def __init__(self, input_nodes=1, hidden_nodes=10, output_nodes=1, learning_rate=0.01):
        # set number of nodes in each input, hidden, output layer
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # weight matrices
        # wih = weights from input(i) to hidden(h)
        # whh = weights from hidden(h) to hidden(h)
        # who = weights from hidden(h) to output(o)
        self.wih = np.random.uniform(0, 1, (self.hidden_nodes, self.input_nodes))
        self.whh = np.random.uniform(0, 1, (self.output_nodes, self.hidden_nodes))
        self.who = np.random.uniform(0, 1, (self.hidden_nodes, self.hidden_nodes))

        # learning rate
        self.learn = learning_rate

    def forward(self, input):
        """ returns a list of hidden states for every timestep
            and the output after a forward pass """
        hidden_states = []
        # initial hidden state is a matrix of zeros
        hidden_states.append(np.zeros((self.hidden_nodes, 1)))
        # forward pass for all timesteps t
        for t in range(input.shape[0]):
            # get next hidden state
            next_hidden_state = np.tanh(np.dot(self.wih, input[[t]].T) + np.dot(self.who, hidden_states[-1]))
            # save next hidden state
            hidden_states.append(next_hidden_state)
        # output from forward pass
        hidden_output = np.dot(self.whh, hidden_states[-1])
        return hidden_states, hidden_output

    # updates the weights
    def update_weights(self, weight_dervs):
        self.wih -= self.learn * weight_dervs["wih"]
        self.whh -= self.learn * weight_dervs["whh"]
        self.who -= self.learn * weight_dervs["who"]

    def error(self, actual, prediction):
        return mse(actual, prediction)

    # backpropagation through time
    def backpropagation(self, input, target, hidden_states, hidden_output):
        weight_dervs = {
                    "wih" : np.zeros_like(self.wih),
                    "who" : np.zeros_like(self.who)
                    }
        # mse
        error = self.error(target, hidden_output)

        weight_dervs["whh"] = np.dot((hidden_output - target), hidden_states[-1].T)

        # gradient of error with respect to whh
        gradient_error = np.dot(self.whh.T, error)

        # gradient of tanh with respect to hidden state
        gradient_hidden_states = gradient_error * d_tanh(hidden_states[-1])

        for t in reversed(range(input.shape[0])):
            weight_dervs["who"] += np.dot(gradient_hidden_states, hidden_states[t-1].T)
            weight_dervs["wih"] += np.dot(gradient_hidden_states, input[[t-1]])

        return weight_dervs

    def train(self, input, target, epochs=100):
        for epoch in range(epochs):
            if epoch == epochs - 1:
                train_outputs = []
            for i in range(input.shape[0]):
                hidden_states, hidden_output = self.forward(input[i])

                # to measure training accuracy
                if epoch == epochs - 1:
                    train_outputs.append(hidden_output.tolist()[0])

                weight_dervs = self.backpropagation(input[i], target[i], hidden_states, hidden_output)

                # update original weights using gradients calculated in backpropagation
                self.update_weights(weight_dervs)

        train_outputs = np.array(train_outputs).T[0]
        return train_outputs

    def test(self, input, target):
        test_outputs = []
        # forward pass for every timestep
        for i in range(input.shape[0]):
            hidden_states, hidden_output = self.forward(input[i])
            test_outputs.append(hidden_output.tolist()[0])

        test_outputs = np.array(test_outputs).T[0]
        return test_outputs

def d_tanh(x):
    return 1 - np.square(np.tanh(x))

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

def plot(actual, train, test):
    plt.plot(actual, label="Actual")
    plt.plot(train, label="Train prediction")
    # x values for test prediction plot
    plt.plot([x for x in range(train.shape[0], train.shape[0] + test.shape[0])], test, label="Test prediction")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    df = get_stock_data("TSLA")
    stock = df["Adj Close"].values

    # normalize
    scaler = Normalize(stock)
    normalized = scaler.normalize_data(stock)

    # get training and testing inputs and outputs
    train_inputs, train_targets, test_inputs, test_targets = train_test_split(normalized)

    # returns 3d array in format [inputs, timesteps, features]
    train_inputs = to_3d(train_inputs)
    test_inputs = to_3d(test_inputs)

    #print(train_inputs.shape, train_targets.shape)
    #print(test_inputs.shape, test_targets.shape)

    NN = RNN()
    train_outputs = NN.train(train_inputs, train_targets, epochs=100)
    test_outputs = NN.test(test_inputs, test_targets)

    # de-normalize
    train_outputs = scaler.denormalize_data(train_outputs)
    train_targets = scaler.denormalize_data(train_targets)
    test_outputs = scaler.denormalize_data(test_outputs)
    test_targets = scaler.denormalize_data(test_targets)

    print("\nTraining MSE Accuracy: {:.4f}%".format(100 - mse(train_targets, train_outputs)))
    print("Training RMSE Accuracy: {:.4f}%".format(100 - rmse(train_targets, train_outputs)))
    print("Training MAPE Accuracy: {:.4f}%".format(100 - mape(train_targets, train_outputs)))

    print("\nTest MSE Accuracy: {:.4f}%".format(100 - mse(test_targets, test_outputs)))
    print("Test RMSE Accuracy: {:.4f}%".format(100 - rmse(test_targets, test_outputs)))
    print("Test MAPE Accuracy: {:.4f}%".format(100 - mape(test_targets, test_outputs)))
    
    # plot the results compared to the original stock data
    plot(stock, train_outputs, test_outputs)

if __name__ == "__main__":
    main()