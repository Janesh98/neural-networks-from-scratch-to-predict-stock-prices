import numpy as np

class LSTM:

    def __init__(self, input = 2, lstm_cell_weights = 2, output = 1, learning_rate = 0.34):
        # set number of nodes in each input, hidden, output layer
        self.input_nodes = input
        self.lstm_cell_weights = lstm_cell_weights
        self.output_nodes = output
        
        # weight matrices
        # wff = weights for forget gate
        # wfi = weights for input gate
        # wfo = weights for ouput gate
        # wfc = weights for candidate
        # who = weights from LSTM cells to output
        self.wff = np.random.randn(self.input_nodes, self.lstm_cell_weights).T
        self.wfi = np.random.randn(self.input_nodes, self.lstm_cell_weights).T
        self.wfo = np.random.randn(self.input_nodes, self.lstm_cell_weights).T
        self.wfc = np.random.randn(self.input_nodes, self.lstm_cell_weights).T
        self.who = np.random.randn(2, 1).T

        # set default LSTM cell state
        self.cell_state = [[1, 1] for i in range(100)]
        self.cell_state = np.array(self.cell_state, dtype=float)
        self.cell_state = np.array(self.cell_state, ndmin=2).T

        # learning rate
        self.learn = learning_rate

    # sigmoid activation function
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    # tanh activation function
    def tanh(self, x):
        return 1 - np.square(np.tanh(x))

    def forget_gate(self, gate_input, h_t_1=1):
        # dot product of input and forget gate weights
        gate_input = np.dot(self.wff, gate_input)
        # multiply by previous cell ouutput
        gate_input = h_t_1 * gate_input
        # apply sigmoid activation function
        gate_output = self.sigmoid(gate_input)
        # update current cell state
        self.cell_state = self.cell_state * gate_output


    def input_gate(self, gate_input, h_t_1=1):
        # dot product of input and input gate weights
        gate_input_1 = np.dot(self.wfi, gate_input)
        # multiply by previous cell ouutput
        gate_input_1 = h_t_1 * gate_input_1
        # dot product of input and candicate gate weights
        gate_input_2 = np.dot(self.wfc, gate_input)
        # multiply by previous cell ouutput
        gate_input_2 = h_t_1 * gate_input_2
        # input gate output
        gate_output = self.sigmoid(gate_input_1) * self.tanh(gate_input_2)
        # update current cell state
        self.cell_state = self.cell_state + gate_output


    def output_gate(self, gate_input, h_t_1=1):
        # dot product of input and output gate weights
        gate_input = np.dot(self.wfo, gate_input)
        # multiply by previous cell ouutput
        gate_input = h_t_1 * gate_input
        # apply sigmoid activation function
        gate_output = self.sigmoid(gate_input)
        # compute cell output
        h_t_1 = self.tanh(self.cell_state) * gate_output

        return h_t_1

        
    def forward(self, input_1, input_2, input_3):
        # starting cell state for first cell
        self.cell_state = [[1, 1] for i in range(len(input_1[0]))]
        self.cell_state = np.array(self.cell_state, dtype=float)
        self.cell_state = np.array(self.cell_state, ndmin=2).T

        # Pass input though first lstm cell
        self.forget_gate(input_1)
        self.input_gate(input_1)
        h_t = self.output_gate(input_1)

        # Pass input though second lstm cell
        self.forget_gate(input_2, h_t)
        self.input_gate(input_2, h_t)
        h_t = self.output_gate(input_2, h_t)

        # Pass input though third lstm cell
        self.forget_gate(input_3, h_t)
        self.input_gate(input_3, h_t)
        h_t = self.output_gate(input_3, h_t)

        # dot product of final cell output and output weights
        final_input = np.dot(self.who, h_t)

        # compute the neural networks output
        final_output = self.sigmoid(final_input)

        return final_output, h_t

    def error(self, target, final_output):
        # output layer error is the (target - actual)
        output_error = target - final_output
        # hidden layer error is the output_error, split by weights, recombined at hidden nodes
        hidden_error = np.dot(self.who.T, output_error) 

        return output_error, hidden_error

    def backpropagation(self,training_input_1, training_input_2, training_input_3, h_t, final_output, output_error, cell_error):
        # update the weights between cells and output
        self.who += self.learn * np.dot((output_error * final_output * (1.0 - final_output)), h_t.T)

        # update the weights within LSTM cells
        self.wff += self.learn * np.dot((cell_error * h_t * (1.0 - h_t)), training_input_1.T)
        self.wfi += self.learn * np.dot((cell_error * h_t * (1.0 - h_t)), training_input_2.T)
        self.wfc += self.learn * np.dot((cell_error * h_t * (1.0 - h_t)), training_input_2.T)
        self.wfo += self.learn * np.dot((cell_error * h_t * (1.0 - h_t)), training_input_3.T)


    def train(self, training_input_1, training_input_2, training_input_3, target):
        # convert lists to 2d arrays
        training_input_1 = np.array(training_input_1, ndmin=2).T
        training_input_2 = np.array(training_input_2, ndmin=2).T
        training_input_3 = np.array(training_input_3, ndmin=2).T
        target = np.array(target, ndmin=2).T

        # forward propagation 
        final_output, h_t = self.forward(training_input_1, training_input_2, training_input_3)

        # calculate output and cell output error
        output_error, cell_error = self.error(target, final_output)

        # back propagation
        self.backpropagation(training_input_1, training_input_2, training_input_3, h_t, final_output, output_error, cell_error)

        return final_output

    def test(self, testing_input_1, testing_input_2, testing_input_3):
        # transpose input
        testing_input_1 = testing_input_1.T
        testing_input_2 = testing_input_2.T
        testing_input_3 = testing_input_3.T
        # forward propagation
        final_output, h_t = self.forward(testing_input_1, testing_input_2, testing_input_3)
        #return final input
        return final_output