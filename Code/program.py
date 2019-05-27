import neuralnetwork as nn
import numpy as np

# Importing the data


def import_data(path):
    '''
    Function for importing the data from the data file
    ----------------------------------------------------
    Inputs
    ----------------------------------------------------
    path: Path to the data file (absolute or relative)
    ----------------------------------------------------
    Outputs
    ----------------------------------------------------
    input_data: The data to be inputed to the Neural network
    output_data: The labels of the input data
    '''

    data = np.loadtxt(path)
    input_data = data[:, :2]
    output_data = data[:, -1]

    return input_data, output_data


def get_predictions(data_in, network):
    '''
    Get a prediction vector using forward propagation
    --------------------------------------------------
    Inputs:
    --------------------------------------------------
    data_in: Data to be used for forward propagation
    network: Neural Network to be used
    --------------------------------------------------
    Output:
    --------------------------------------------------
    np.array(predictions): vector of outputs of the NN
    '''
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
    layer_sizes = (2, 20, 1)
    net = nn.NeuralNetwork(layer_sizes)
    # Get predictions from forward propagation
    predictions = get_predictions(data_in, net)
