import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        weight_shapes = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]
        print(weight_shapes)
        self.weights = [np.random.standard_normal(s) / s[1]**.5 for s in weight_shapes]
        self.biases = [np.ones((s, 1)) for s in layer_sizes[1:]]

    def predict(self, a):
        iter = 0
        for w, b in zip(self.weights, self.biases):
            a = np.matmul(w, a) + b
            print(a)
            iter += 1
            print(iter)
        return a

    @staticmethod
    def activation(x):
        return 1/(1+np.exp(-x))
