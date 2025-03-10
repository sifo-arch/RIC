import numpy as np

"""
A Multi Layer perceptron For Binary Classification (signle output neuron)
"""
class BinNeuralNetwork:
    def __init__(self, hidden_layers: np.ndarray, X: np.ndarray, y: np.ndarray):
        """
        PARAMETERS:
        hidden_layers: an array of integers, its length is the number of hidden layers (excluding input and output layers), and each integer in the index i represents the number of neurons in the layer i
        X: data
        y: labels
        """
        # store the dataset
        self._X = X
        self._y = y

        # extend the array of hidden layers to include the output layer (1 output neuron)
        layers = np.append(hidden_layers, 1)

        # store the number of layers
        self._number_of_layers = layers.size

        # the following code generates weights of each layer randomly according to Xavier technique which depends on gaussian distribution
        # weights are stored in a list of arrays, each element in the index i represents a 2d array of wights between layer i and i+1
        
        weights = []
        fan_in = X.shape[1]  # the size of input
        fan_out = layers[0]  # the number of neurons of the first hidden layer
        std = np.sqrt(2 / (fan_in + fan_out))  # standard deviation according to Xavier technique
        W = np.random.normal(0.0, std, (fan_out, fan_in))  # number of neurons = number of lines, number of input = number of columns
        weights.append(W)
        for i in range(1, layers.size):
            fan_in = layers[i - 1]
            fan_out = layers[i]
            std = np.sqrt(2 / (fan_in + fan_out))
            W = np.random.normal(0.0, std, (fan_out, fan_in))
            weights.append(W)
        self._weights = weights

        # initialize biases as zeros
        # biases are stored in a list of arrays, each element in the index i represents a vector of biases of layer i+1
        self._biases = []
        for k in layers:
            biases_i = np.zeros(k).reshape(k, 1)  # column vector
            self._biases.append(biases_i)

        # tracing...
        # print("INITIAL WEIGHTS:")
        # for i, (w, b) in enumerate(zip(self._weights, self._biases)):
        #     print(f"weights of layer [{i + 1}]:")
        #     print(w)
        #     print(f"biases of layer [{i + 1}]:")
        #     print(b)
    

    # the name of methods below is enoughly expressive
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self):
        X = self._X
        W = self._weights[0]
        b = self._biases[0]
        potential = W.dot(X.T) + b
        Z = self._sigmoid(potential)
        for i in range(1, self._number_of_layers):
            W = self._weights[i]
            b = self._biases[i]
            potential = W.dot(Z) + b
            Z = self._sigmoid(potential)
        return Z




X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 0])
layers = np.array([5, 5])
model = BinNeuralNetwork(layers, X, y)
z = model.forward_propagation()
print(z)
