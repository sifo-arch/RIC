import numpy as np
from tqdm import tqdm


class BinNeuralNetwork:
    """
    A Multi Layer perceptron For Binary Classification (signle output neuron).
    It implements Xavier initialization to intialize weights.
    """
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

        # the following code generates weights of each layer randomly according to Xavier initialization which depends on gaussian distribution
        # weights are stored in a list of arrays, each element in the index i represents a 2d array of wights between layer i and i+1
        
        weights = []
        fan_in = X.shape[1]  # the size of input
        fan_out = layers[0]  # the number of neurons of the first hidden layer
        std = np.sqrt(2 / (fan_in + fan_out))  # standard deviation according to Xavier technique
        W = np.random.normal(0.0, std, (fan_out, fan_in))  # number of neurons = number of lines, number of inputs = number of columns
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
    

    @property
    def weights(self):
        """
        weights getter
        """
        return self._weights
    

    @property
    def biases(self):
        """
        biases getter
        """
        return self._biases
    

    def _sigmoid(self, x):
        """
        sigmoid as activation function
        """
        return 1 / (1 + np.exp(-x))
    

    def forward_propagation(self, X):
        """
        This method performs forward propagation on the X argument.
        X could be one datapoint or a set of datapoints.
        """
        # retrieve the weights and biases of the first layer
        W = self._weights[0]
        b = self._biases[0]
        # if X is a vector (single datapoint) make b a line-vector (for dimensions compatibality)
        if X.ndim == 1:
            b = b.flatten()
        # calculate potential and activation of the first layer
        potential = W.dot(X.T) + b
        Z = self._sigmoid(potential)
        # tracing...
        print(f"Activations of layer[{1}]:")
        print(Z)
        # repeat the process for the rest of layers layers
        for i in range(1, self._number_of_layers):

            W = self._weights[i]
            b = self._biases[i]

            if X.ndim == 1:
                b = b.flatten()
            
            potential = W.dot(Z) + b
            Z = self._sigmoid(potential)
            
            # tracing
            print(f"Activations of layer[{i + 1}]:")
            print(Z)
        
        # return the output of the last layer (output layer)
        return Z
    
    
    def forward_outputs(self):
        """
        This method performs forward propagation on the entire dataset.
        However, it saves the output of each layer in a list and returns
        the list.
        """
        outputs = []

        X = self._X
        W = self._weights[0]
        b = self._biases[0]
        potential = W.dot(X.T) + b
        Z = self._sigmoid(potential)

        outputs.append(Z)

        for i in range(1, self._number_of_layers):
            W = self._weights[i]
            b = self._biases[i]
            potential = W.dot(Z) + b
            Z = self._sigmoid(potential)

            outputs.append(Z)
        
        return outputs
    

    def calculate_derivatives(self, outputs):
        """
        This method calculates derivatives of each layer.

        PARAMETERS:
        outputs: a list, each element in index i is the output of layer i+1
        """
        # A list to store derivatives
        derivatives = []
        
        # At First: Calculate the Derivative of the Last Layer
        # last output (transpose it here to optimize the code)
        loT = outputs[-1].T

        # adjust dimensions of y
        y = self._y.reshape((self._y.size, 1))
        # last layer derivative
        lld = loT * (1 - loT) * (y - loT)

        # push lld to the list
        derivatives.insert(0, lld)

        # Next: Calculate the rest of layers' derivatives
        n = self._number_of_layers
        for i in range(n - 2, -1, -1):
            # output of layer i+1
            Oi = outputs[i]
            # derivative of the next layer
            nd = derivatives[0]
            # weights between layer i+1 and i+2
            W = self._weights[i + 1]
            # calculate the derivative
            d = (Oi * (1 - Oi)).T * (nd.dot(W))

            # push d to the list
            derivatives.insert(0, d)
        
        return derivatives


    def update_weights(self, alpha, derivatives, outputs):
        """
        This method makes one update to weights of the entire network.
        PARAMETERS:
        alpha: learning rate
        derivatives: a list of arrays, each element in the index [i] is a 2d array containing derivatives of the layer i+1
        outputs:  a list of arrays, each element in index [i] is 2d array containing outputs of layer i+1
        """
        # weights between the input and the first layer
        W = self._weights[0]
        # derivatives of the first layer (transpose)
        dt = derivatives[0].T
        # the dataset (outputs of layer 0)
        O = self._X

        # update weights between the input and the first layer
        self._weights[0] = W + alpha * (dt.dot(O))

        # repeat the process for the rest of weights
        n = self._number_of_layers
        for i in range(1, n):
            # weights between layer i and layer i+1
            W = self._weights[i]
            # derivatives of layer i+1
            dt = derivatives[i].T
            # outputs of layer i (transpose for dimensions compatibility)
            O = outputs[i - 1].T

            # update weights between layer i and layer i+1
            self._weights[i] = W + alpha * (dt.dot(O))
    

    def update_biases(self, alpha, derivatives):
        """
        This method makes one update to biases of the entire network
        PARAMETERS:
        The same as update_weights method except outputs
        """
        # number of dataset examples
        T = self._X.shape[0]
        # an array of ones reshaped to be compatible with the use of matrix multiplication in order to simplify the implementation
        ones = np.ones((T, 1))
        # number of layers
        n = self._number_of_layers
        for i in range(n):
            # old biases of the layer i+1
            b = self._biases[i]
            # derivative of layer i+1 (transpose)
            dt = derivatives[i].T

            # update biases of layer i+1
            self._biases[i] = b + alpha * (dt.dot(ones))

    
    def train(self, alpha, iter):
        for _ in tqdm(range(iter)):
            outputs = self.forward_outputs()
            derivatives = self.calculate_derivatives(outputs)
            self.update_weights(alpha, derivatives, outputs)
            self.update_biases(alpha, derivatives)
            






# Dataset (XOR)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
# Labels (XOR)
y = np.array([0, 1, 1, 0])
# Architecture
layers = np.array([3, 2])
# model instantiation
model = BinNeuralNetwork(layers, X, y)


# # test forward propagation on the dataset
# print("Forward propagation of the dataset:")
# z_X = model.forward_propagation(X)

# # test forward propagation on a new datapoint
# xi = np.array([1, 0])
# print(f"Forward propagation of {xi}:")
# z_xi = model.forward_propagation(xi)

print("Initial weights:")
for w in model.weights:
    print(w)
    print()

print("Initial biases:")
for b in model.biases:
    print(b)
    print()

# # test forward_outputs
# print("Forward outputs:")
# outputs = model.forward_outputs()
# for o in outputs:
#     print(o)
#     print()

# # test calculate_derivatives
# print("Derivatives:")
# derivatives = model.calculate_derivatives(outputs)
# for d in derivatives:
#     print(d)
#     print()

# learning rate
alpha = 0.1

# # test update_weights
# print("Updated weights:")
# model.update_weights(alpha, derivatives, outputs)
# for w in model.weights:
#     print(w)
#     print()

# # test update_biases
# print("Updated biases:")
# model.update_biases(alpha, derivatives)
# for b in model.biases:
#     print(b)
#     print()

# number of iterations
iter = 10000

# train the network
model.train(alpha, iter)

print("Final weights:")
for w in model.weights:
    print(w)
    print()

print("Final biases:")
for b in model.biases:
    print(b)
    print()


# test the network on the dataset
predictions = model.forward_propagation(X)
# print(predictions)



