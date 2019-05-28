import numpy as np


class NeuralNetwork:
    '''
    Creates a Neural Network with a structure defined by
    layer_sizes
    '''

    def __init__(self, layer_sizes):
        '''
        Initializes the Neural Network weights and biases with random values
        weight_shapes: The shapes for every weight matrix
        self.weights: Initialiazation of the weight matrices for each layer based on a standard_normal distribution
        self.biases: Initialization of the biases for each layer with ones
        self.layer_sizes: Tuple containing the size of each layer
        '''
        weight_shapes = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.weights = [np.random.standard_normal(s) / s[1]**.5 for s in weight_shapes]
        self.biases = [np.ones((s, 1)) for s in layer_sizes[1:]]
        self.layer_sizes = layer_sizes

    def forward(self, a):
        '''
        Calculates the forward progpa
        '''
        for w, b in zip(self.weights, self.biases):
            a = self.activation(np.matmul(w, a) + b)
        return a

    def loss_function(self, labels):
        pass

    @staticmethod
    def activation(x):
        return 1 / (1 + np.exp(-x))
