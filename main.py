import numpy as np
from mlp import BinNeuralNetwork

FILE_PATH = "data.txt"


def prepare_dataset(file_path):
    """
    This function prepares the dataset as X (data), y (labels)
    PARAMETERS:
    file_path: the path of file containing the dataset
    """
    # to store the dataset
    X = []
    y = []
    # open the file (mode=read)
    with open(file_path, "r") as f:
        # iterate over the list of lines
        for raw_line in f.readlines():
            # remove begin/end spaces and EOF character, then split the string by spaces
            line = raw_line.strip().split(r" ")
            # parse line to a list of floats (data), and integer
            xi = [float(e) for e in line[:-1]]
            yi = int(line[-1])
            # append xi and yi to the dataset
            X.append(xi)
            y.append(yi)
        
    return np.array(X), np.array(y)


def normalize_data(X):
    """
    This function performs in place normalization of a dataset
    """
    # number of columns
    cols = range(X.shape[1])
    # an array contains max value of each column
    max = np.array([np.max(X[:, i]) for i in cols])
    # an array contains min value of each column
    min = np.array([np.min(X[:, i]) for i in cols])

    # normalize the data
    for j in cols:
        X[:, j] = (X[:, j] - min[j]) / (max[j] - min[j])




# prepare the dataset
X, y = prepare_dataset(FILE_PATH)
# normaliza the set of data
normalize_data(X)
# model architecture
architecture = [5, 5, 5, 5, 5]
# instantiate the model
model = BinNeuralNetwork(architecture, X, y)

# hyper-parameters
alpha = 0.1
iter = 10000

# train the model
model.train(alpha, iter)

# model score on the dataset
score = model.score(X, y)
print(f"Model score on the dataset: {score}")

