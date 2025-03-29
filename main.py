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


# prepare the dataset
X, y = prepare_dataset(FILE_PATH)
# model architecture
architecture = [50, 50, 50]
# instantiate the model
model = BinNeuralNetwork(architecture, X, y)

# hyper-parameters
alpha = 0.01
iter = 100000

# train the model
model.train(alpha, iter)

# model score on the dataset
score = model.score(X, y)
print(f"Model score on the dataset: {score}")

