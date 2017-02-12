import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(sigmoid_x):
    return sigmoid_x * (1 - sigmoid_x)

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                       (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5,
                                       (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate

        # Activation function is the sigmoid function
        self.activation_function = sigmoid

    def forwardpass(self, inputs_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        hidden_outputs = sigmoid(hidden_inputs)

        # Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = final_inputs # Activation function for this node is f(x) = x

        return hidden_inputs, hidden_outputs, final_inputs, final_outputs

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        ### Forward pass ###
        hidden_inputs, hidden_outputs, final_inputs, final_outputs = self.forwardpass(inputs_list)

        ### Backward pass ###
        # Output error
        output_errors = targets - final_outputs

        # Backpropagated error
        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors)
        hidden_grad = hidden_errors * hidden_outputs * (1 - hidden_outputs)

        # Update the weights
        self.weights_hidden_to_output += (self.lr * output_errors * hidden_outputs).T
        self.weights_input_to_hidden += (self.lr * hidden_grad * inputs.T)


    def run(self, inputs_list):
        # Run a forward pass through the network
        hidden_inputs, hidden_outputs, final_inputs, final_outputs = self.forwardpass(inputs_list)
        return final_outputs
