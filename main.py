import numpy as np
import matplotlib.pyplot as plt
from mlp import BinNeuralNetwork


FILE_PATH = "data.txt"
TEST_PATH = "test.txt"
WEIGHTS_FOLDER = "saved_weights"
WEIGHTS_FILE = "vase.dat"


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


def filter_predicted_vase(X, predictions):
    """
    This function extracts points that are predicted to belong to the vase

    PARAMETERS:
    X: dataset (points)
    predictions: array of dataset model predictions

    RETURNS:
    a 2d array of points that are predicted to belong to the vase
    """
    # a list to store points predicted to belong to the vase
    X_vase = []
    # for each prediction
    for i in range(predictions.size):
        # if prediction is belong to the vase (1)
        if predictions[i] == 1:
            # append the corresponding point to the list
            X_vase.append(X[i])
    
    # return the list of point as an array
    return np.array(X_vase)


def plot_predicted_vase(X_vase):
    """
    This function makes a 3d plot of points that are belong to the vase (AI assisted)
    PARAMETERS:
    X_vase: points that are belong to the vase
    """
    # extracts points coordinations
    x, y, z = X_vase[:, 0], X_vase[:, 1], X_vase[:, 2]

    # 3d scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')

    # axes labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # show the plot
    plt.show()


def training_procedure(model: BinNeuralNetwork, save_weights=False):
    """
    This procedure trains the model, and can save resulted weights
    PARAMETERS:
    model: the model
    save_weights: a boolean, if True weights (resulted from training) will be saved
    """
    # hyper-parameters
    alpha = 0.0002
    iter = 10000

    # train the model
    model.train(alpha, iter, track_loss=True)

    # # modify hyper-parameters
    # alpha = 0.00028
    # iter = 15000
    # # train the model again
    # model.train(alpha, iter, track_loss=True)

    # # modify hyper-parameters
    # alpha = 0.00025
    # iter = 15000
    # # train again
    # model.train(alpha, iter, track_loss=True)

    if save_weights:
        path = WEIGHTS_FOLDER + "/" + WEIGHTS_FILE
        model.save_weights(path)


def load_test_set(file_path):
    X = []
    with open(file_path, mode='r') as f:
        for raw_line in f.readlines():
            line_elements = raw_line.split(r" ")
            line_values = [float(e) for e in line_elements]
            X.append(line_values)
    return np.array(X)


def write_test_labels(file_path, X_test, predictions):
    with open(file_path, mode='w') as f:
        for i in range(X_test.shape[0]):
            str_line = " ".join([str(e) for e in X_test[i]]) + " " + str(int(predictions[i]))
            str_line += '\n'
            f.write(str_line)


# prepare the dataset
X, y = prepare_dataset(FILE_PATH)
# normaliza the set of data
# normalize_data(X)
# model architecture
architecture = [3, 3]
# instantiate the model
model = BinNeuralNetwork(architecture, X, y)

# # train the model and save weights
# training_procedure(model, save_weights=True)

# load ready weights
path = WEIGHTS_FOLDER + "/" + WEIGHTS_FILE
model.load_weights(path) # 97.76 % accuracy (20000 points)

# model predictions and score on the dataset
predictions, score = model.score(X, y)
print(f"Model score on the dataset: {score}")

# points predicted to belong to the vase
X_vase = filter_predicted_vase(X, predictions)

# plot X_vase
plot_predicted_vase(X_vase)

# =========================================================
X_test = load_test_set(TEST_PATH)
X_test_pred = model.predict(X_test)
X_test_vase = filter_predicted_vase(X_test, X_test_pred)
plot_predicted_vase(X_test_vase)
NEW_TEST_PATH = "new_test.txt"
write_test_labels(NEW_TEST_PATH, X_test, X_test_pred)
