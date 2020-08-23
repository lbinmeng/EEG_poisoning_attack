import scipy.io as scio
import numpy as np
from .csp import CSP


def mi_load(data_path, s_id, is_processed=None):
    """ load MI dataset """
    data_temp = scio.loadmat(data_path + '/A0' + str(s_id+1) + '.mat')
    data = np.transpose(data_temp['x'], (2, 0, 1))
    labels = np.asarray(data_temp['y']).squeeze()

    if is_processed == 'cov':
        data = cov_process(data)
        data = np.reshape(data, [len(data), -1])
    elif is_processed == 'csp':
        data = csp_process([data, labels], filter)
        data = np.reshape(data, [len(data), -1])
    else:
        data = np.reshape(data, [data.shape[0], 1, data.shape[1], data.shape[2]])

    return data, labels


def cov_process(data):
    """ Covariance matrix """
    cov_data = []
    data_size = len(data)
    for i in range(data_size):
        data_temp = np.dot(data[i], np.transpose(data[i]))  # / np.trace(np.dot(data[i], np.transpose(data[i])))
        data_temp = np.reshape(data_temp, [-1])
        cov_data.append(data_temp)

    return np.asarray(cov_data)


def csp_process(data, filter):
    """ Common Spatial Pattern """
    csp_data = []
    data_size = len(data[0])
    for i in range(data_size):
        data_temp = np.dot(filter, data[0][i])
        data_temp = np.dot(data_temp, np.transpose(data_temp)) / np.trace(
            np.dot(data_temp, np.transpose(data_temp)))
        csp_data.append(data_temp)

    return np.asarray(csp_data)
