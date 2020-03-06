import numpy as np

class FeedForward:

    def __init__(self, input = 2, hidden = 3, output = 1, learning_rate = 0.3):
        # number of nodes for each layer
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
        # matrix dot product of weight wih and input
        # to produce the input for the hidden layer
        hidden_input = np.dot(self.wih, input)
        # squash hidden_input into range (0, 1)
        # to produce output of hidden layer
        hidden_output = self.sigmoid(hidden_input)
        
        # matrix dot product of weight who and output from
        # hidden layer to produce the input for the output layer
        final_input = np.dot(self.who, hidden_output)

        # squash final_input into range (0, 1)
        # to produce output of output layer
        final_output = self.sigmoid(final_input)

        return final_output, hidden_output

    def error(self, target, final_output):
        # error is the distance from target and prediction
        error = target - final_output
        # hidden error is the dot product of the weight who transposed
        # and the error calcualted above
        hidden_error = np.dot(self.who.T, error) 

        return error, hidden_error

    def backpropagation(self, input, hidden_output, final_output, error, hidden_error):
        # update the weight who with errors previously calculated
        self.who += self.learn * np.dot((error * final_output * (1.0 - final_output)), hidden_output.T)
        
        # update the weight wih with errors previously calculated
        self.wih += self.learn * np.dot((hidden_error * hidden_output * (1.0 - hidden_output)), input.T)
        
    def train(self, input, target):
        # reshape input and target into 2d matrices
        input = np.array(input, ndmin=2).T
        target = np.array(target, ndmin=2).T

        # forward pass throught network
        final_output, hidden_output = self.forward(input)

        # get errors from forward pass result
        error, hidden_error = self.error(target, final_output)

        # backpropagate the errors through the network, updating weights
        self.backpropagation(input, hidden_output, final_output, error, hidden_error)

        # return training results
        return final_output

    def test(self, input):
        # transpose input
        input = input.T
        # perform one forward pass through the network to get predictions
        final_output, hidden_output = self.forward(input)

        # return prediction
        return final_output