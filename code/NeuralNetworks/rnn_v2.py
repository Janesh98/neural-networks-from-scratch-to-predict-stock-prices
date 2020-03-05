import numpy as np

class RNN_V2():
    def __init__(self, input_nodes=1, hidden_nodes=10, output_nodes=1, learning_rate=0.01):
        # set number of nodes in each input, hidden, output layer
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # weight matrices
        # wih = weights from input(i) to hidden(h)
        # whh = weights from hidden(h) to hidden(h)
        # who = weights from hidden(h) to output(o)
        self.wih = np.random.uniform(0, 1, (self.hidden_nodes, self.input_nodes)) / 2
        self.whh = np.random.uniform(0, 1, (self.output_nodes, self.hidden_nodes)) / 2
        self.who = np.random.uniform(0, 1, (self.hidden_nodes, self.hidden_nodes)) / 2

        # learning rate
        self.learn = learning_rate

    def d_tanh(self, x):
        return 1 - np.square(np.tanh(x))

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
    def update_weights(self, wih, whh, who):
        self.wih -= self.learn * wih
        self.whh -= self.learn * whh
        self.who -= self.learn * who

    def error(self, actual, prediction):
        return np.mean(np.square(actual - prediction))

    # backpropagation through time
    def backpropagation(self, input, target, hidden_states, hidden_output):
        # mse
        error = self.error(target, hidden_output)

        wih = np.zeros_like(self.wih)
        whh = np.dot((hidden_output - target), hidden_states[-1].T)
        who = np.zeros_like(self.who)

        # gradient of error with respect to whh
        gradient_error = np.dot(self.whh.T, error)

        # gradient of tanh with respect to hidden state
        gradient_hidden_states = gradient_error * self.d_tanh(hidden_states[-1])

        for t in reversed(range(input.shape[0])):
            who += np.dot(gradient_hidden_states, hidden_states[t-1].T)
            wih += np.dot(gradient_hidden_states, input[[t-1]])

        # update original weights using gradients calculated in backpropagation
        self.update_weights(wih, whh, who)

    def train(self, input, target, epochs=100):
        for epoch in range(epochs):
            if epoch == epochs - 1:
                train_outputs = []
            for i in range(input.shape[0]):
                hidden_states, hidden_output = self.forward(input[i])

                # to measure training accuracy
                if epoch == epochs - 1:
                    train_outputs.append(hidden_output.tolist()[0])

                self.backpropagation(input[i], target[i], hidden_states, hidden_output)

        train_outputs = np.array(train_outputs).T[0]
        return train_outputs

    def test(self, input):
        test_outputs = []
        # forward pass for every timestep
        for i in range(input.shape[0]):
            hidden_states, hidden_output = self.forward(input[i])
            test_outputs.append(hidden_output.tolist()[0])

        test_outputs = np.array(test_outputs).T[0]
        return test_outputs