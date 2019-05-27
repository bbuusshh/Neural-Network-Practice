import neuralnetwork as nn
import numpy as np

# Importing the data


def import_data(filename):
    data = np.loadtxt(filename)
    input_data = data[:, :2]
    output_data = data[:, -1]

    return input_data, output_data


def get_predictions(data_in, network):
    predictions = []

    for data in data_in:
        # Convert data from row to column vector
        data = np.reshape(data, (-1, 1))
        predictions.append(network.forward(data))

    # Format the data
    for i in range(len(predictions)):
        predictions[i] = np.asscalar(predictions[i])

    return np.array(predictions)


if __name__ == '__main__':

    filename = 'data2Class.txt'

    data_in, data_out = import_data(filename)

    # Initialize the Neural Network
    layer_sizes = (2, 1)
    net = nn.NeuralNetwork(layer_sizes)
    # Get predictions from forward propagation
    predictions = get_predictions(data_in, net)
    print(predictions.shape)
