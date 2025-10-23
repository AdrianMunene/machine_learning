import numpy as np 
import gzip
import struct

def load_images(filename):

    with gzip.open(filename, 'rb') as f:
        _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))

        all_pixels = np.frombuffer(f.read(), dtype = np.uint8)

        return all_pixels.reshape(n_images, columns * rows)

def prepend_bias(X):
    return np.insert(X, 0, 1, axis = 1)

def load_labels(filename):

    with gzip.open(filename, 'rb') as f:
        f.read(8)

        all_labels = f.read()

        return np.frombuffer(all_labels, dtype = np.uint8).reshape(-1, 1)

def one_hot_encode(Y):
    n_labels = Y.shape[0]
    n_classes = 10
    encoded_Y = np.zeros((n_labels, n_classes))

    for i in range(n_labels):
        label = Y[i]
        encoded_Y[i][label] = 1

    return encoded_Y

def standardize(training_set, test_set):
    average = np.average(training_set)
    standard_deviation = np.std(training_set)

    training_set_standardized = (training_set - average) / standard_deviation

    test_set_standardized = (test_set - average) / standard_deviation

    return (training_set_standardized, test_set_standardized)


#X_train = load_images("../data/mnist/train-images-idx3-ubyte.gz") 

#X_test =  load_images("../data/mnist/t10k-images-idx3-ubyte.gz") 

X_train_raw = load_images("../data/mnist/train-images-idx3-ubyte.gz")
X_test_raw = load_images("../data/mnist/t10k-images-idx3-ubyte.gz")

X_train, X_test_all = standardize(X_train_raw, X_test_raw)
X_validation, X_test = np.split(X_test_all, 2)


Y_train_unencoded = load_labels("../data/mnist/train-labels-idx1-ubyte.gz")
Y_train = one_hot_encode(Y_train_unencoded)

Y_test_all = load_labels("../data/mnist/t10k-labels-idx1-ubyte.gz")
Y_validation, Y_test = np.split(Y_test_all, 2)