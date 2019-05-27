import neuralnetwork as nn
import numpy as np

# Importing the data

layer_sizes = (1, 3, 1)
x = np.array([1])

net = nn.NeuralNetwork(layer_sizes)

for w in net.weights:
    print(w)
for w in net.biases:
    print(w)

pred = net.predict(x)

print(pred)
