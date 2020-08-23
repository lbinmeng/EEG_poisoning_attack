import numpy as np
from scipy.io import loadmat


def cross_data(path):
    """ get cross-subject data """
    data = loadmat(path)
    x_train = data['x_train']
    y_train = np.squeeze(data['y_train'])
    x_test = data['x_test']
    y_test = np.squeeze(data['y_test'])

    return np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0)
