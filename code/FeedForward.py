import numpy as np

class FeedForward:

    def __init__(self, input = 2, hidden = 3, output = 1, learning_rate = 0.3):
        # set number of nodes in each input, hidden, output layer
        self.input_nodes = input
        self.hidden_nodes = hidden
        self.output_nodes = output
        
        # weight matrices
        # wih = weights from input(i) to hidden(h)
        # who = weights from hidden(i) to output(o)
        self.wih = np.random.randn(self.input_nodes, self.hidden_nodes).T
        self.who = np.random.randn(self.hidden_nodes, self.output_nodes).T

        # learning rate
        self.learn = learning_rate

    # sigmoid activation function
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def forward(self, input):
        # calculate signals into hidden layer
        hidden_input = np.dot(self.wih, input)
        # calculate the signals emerging from hidden layer
        hidden_output = self.sigmoid(hidden_input)
        
        # calculate signals into final output layer
        final_input = np.dot(self.who, hidden_output)

        # calculate the signals emerging from final output layer
        final_output = self.sigmoid(final_input)

        return final_output, hidden_output

    def error(self, target, final_output):
        # output layer error is the (target - actual)
        output_error = target - final_output
        # hidden layer error is the output_error, split by weights, recombined at hidden nodes
        hidden_error = np.dot(self.who.T, output_error) 

        return output_error, hidden_error

    def backpropagation(self, input, hidden_output, final_output, output_error, hidden_error):
        # update the weights between hidden and output
        self.who += self.learn * np.dot((output_error * final_output * (1.0 - final_output)), hidden_output.T)
        
        # update the weights between input and hidden
        self.wih += self.learn * np.dot((hidden_error * hidden_output * (1.0 - hidden_output)), input.T)
        
    def train(self, input, target):
        # convert lists to 2d arrays
        input = np.array(input, ndmin=2).T
        target = np.array(target, ndmin=2).T

        final_output, hidden_output = self.forward(input)

        output_error, hidden_error = self.error(target, final_output)

        self.backpropagation(input, hidden_output, final_output, output_error, hidden_error)

        return final_output

    def test(self, input):
        # transpose input
        input = input.T
        final_output, hidden_output = self.forward(input)

        return final_output