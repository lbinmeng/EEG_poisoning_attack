import numpy as np
import lib.utils as utils
from lib import ERNDataget, MIDataget, P300Dataget
import os
from scipy.io import loadmat


def load(data_name, s_id, npp_params, clean=True, physical=False, partial=None, downsample=True):
    """ load ERN data """
    if clean:
        path = 'EEG_Data/' + data_name + '/clean'
    else:
        if physical:
            path = 'EEG_Data/' + data_name + '/physical-poisoned-{}-{}-{}'.format(npp_params[0], npp_params[1],
                                                                                  npp_params[2])
        else:
            path = 'EEG_Data/' + data_name + '/poisoned-{}-{}-{}'.format(npp_params[0], npp_params[1], npp_params[2])

    if partial:
        path = 'EEG_Data/' + data_name + '/partial-{}_poisoned-{}-{}-{}'.format(partial, npp_params[0], npp_params[1],
                                                                                npp_params[2])

    if not os.path.exists(path):
        if data_name == 'ERN':
            ERNDataget.get(npp_params, clean, physical, partial)
        elif data_name == 'MI':
            MIDataget.get(npp_params, clean, physical, partial)
        elif data_name == 'P300':
            P300Dataget.get(npp_params, clean, physical, partial)

    data = loadmat(path + '/s{}.mat'.format(s_id))
    eeg = data['eeg']
    x = data['x']
    y = data['y']
    y = np.squeeze(y.flatten())
    # downsample
    if downsample:
        x1 = x[np.where(y == 0)]
        x2 = x[np.where(y == 1)]
        sample_num = min(len(x1), len(x2))
        idx1, idx2 = utils.shuffle_data(len(x1)), utils.shuffle_data(len(x2))
        x = np.concatenate([x1[idx1[:sample_num]], x2[idx2[:sample_num]]], axis=0)
        y = np.concatenate([np.zeros(shape=[sample_num]), np.ones(shape=[sample_num])], axis=0)

    return eeg, x, y
