import os
import mne
import numpy as np
from scipy.io import loadmat
from utils import ERNDataget, P300Dataget, MI4CDataget
from utils.asr import asr


def shuffle_data(data_size, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    indices = np.arange(data_size)
    return np.random.permutation(indices)


def load(data_name, uid, npp_params, clean=True, physical=False, partial=None, downsample=True, noise_type='npp'):
    """ load ERN data """
    if clean:
        path = f'data/{data_name}/clean/'
    else:
        if physical:
            path = f'data/{data_name}/physical-poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
        else:
            path = f'data/{data_name}/poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
    if partial:
        path = f'data/{data_name}/partial-{partial}_poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
    
    if noise_type != 'npp': path = path.replace('poisoned', f'{noise_type}')

    if not os.path.exists(path):
        if data_name == 'ERN':
            ERNDataget.get(npp_params, clean, physical, partial, noise_type)
        elif data_name == 'P300':
            P300Dataget.get(npp_params, clean, physical, partial, noise_type)
        elif data_name == 'MI4C':
            MI4CDataget.get(npp_params, clean, physical, partial, noise_type)
        else:
            raise Exception(f'No such dataset: {data_name}')


    data = loadmat(path + f'/s{uid}.mat')
    eeg = data['eeg']
    x = data['x']
    y = data['y']
    y = np.squeeze(y.flatten())
    # downsample
    if downsample:
        x1 = x[np.where(y == 0)]
        x2 = x[np.where(y == 1)]
        sample_num = min(len(x1), len(x2))
        idx1, idx2 = shuffle_data(len(x1)), shuffle_data(len(x2))
        x = np.concatenate([x1[idx1[:sample_num]], x2[idx2[:sample_num]]], axis=0)
        y = np.concatenate([np.zeros(shape=[sample_num]), np.ones(shape=[sample_num])], axis=0)

    return eeg, x, y


def surface_laplacian(eeg, data_name):
    ch_path = f'data/{data_name}.txt'
    ch_names = []
    with open(ch_path, 'r') as file:
        for line in file.readlines():
            line = line.replace('\n', '').split('\t')
            ch_names.append(line[-1])
    
    info = mne.create_info(
        ch_names=ch_names,
        ch_types=['eeg'] * len(ch_names),
        sfreq=128
    )
    info.set_montage('standard_1020')
    epochs = mne.EpochsArray(eeg.squeeze(), info)
    epochs_sl = mne.preprocessing.compute_current_source_density(epochs)
    return epochs_sl.get_data()


def average_referencing(eeg):
    eeg_ar = eeg - np.mean(eeg, axis=-2, keepdims=True)
    return eeg_ar


def artifact_subspace_reconstruction(eeg, data_name):
    # ch_path = f'data/{data_name}.txt'
    # ch_names = []
    # with open(ch_path, 'r') as file:
    #     for line in file.readlines():
    #         line = line.replace('\n', '').split('\t')
    #         ch_names.append(line[-1])

    # info = mne.create_info(
    #     ch_names=ch_names,
    #     ch_types=['eeg'] * len(ch_names),
    #     sfreq=128
    # )
    # n, c, s = eeg.shape
    # eeg = np.transpose(eeg, (1, 2, 0)).reshape(c, -1)
    # raw = mne.io.RawArray(eeg, info)
    eeg = eeg.squeeze()
    n, c, s = eeg.shape
    eeg = np.transpose(eeg, (1, 2, 0)).reshape(c, -1)
    pre_cleaned, _ = asr.clean_windows(eeg, sfreq=128, max_bad_chans=0.1)
    M, T = asr.asr_calibrate(pre_cleaned, sfreq=128, cutoff=15)
    clean_eeg = asr.asr_process(eeg, sfreq=128, M=M, T=T)
    clean_eeg = np.transpose(clean_eeg.reshape(c, s, n), (2, 0, 1))
    return clean_eeg[:, np.newaxis, :, :]


def load_with_advanced_processing(data_name, uid, npp_params, clean=True, physical=False, partial=None, downsample=True, noise_type='npp', process='ar'):
    if clean:
        path = f'data/{data_name}/clean/'
    else:
        if physical:
            path = f'data/{data_name}/physical-poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
        else:
            path = f'data/{data_name}/poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
    if partial:
        path = f'data/{data_name}/partial-{partial}_poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
    
    if noise_type != 'npp': path = path.replace('poisoned', f'{noise_type}')
    if process != 'none': path = path[:-1] + f'_{process}/'

    if not os.path.exists(path):
        if data_name == 'ERN':
            ERNDataget.get(npp_params, clean, physical, partial, noise_type, process)
        elif data_name == 'P300':
            P300Dataget.get(npp_params, clean, physical, partial, noise_type, process)
        elif data_name == 'MI4C':
            MI4CDataget.get(npp_params, clean, physical, partial, noise_type, process)
        else:
            raise Exception(f'No such dataset: {data_name}')
    
    data = loadmat(path + f'/s{uid}.mat')
    eeg = data['eeg']
    x = data['x']
    y = data['y']
    y = np.squeeze(y.flatten())
    # downsample
    if downsample:
        x1 = x[np.where(y == 0)]
        x2 = x[np.where(y == 1)]
        sample_num = min(len(x1), len(x2))
        idx1, idx2 = shuffle_data(len(x1)), shuffle_data(len(x2))
        x = np.concatenate([x1[idx1[:sample_num]], x2[idx2[:sample_num]]], axis=0)
        y = np.concatenate([np.zeros(shape=[sample_num]), np.ones(shape=[sample_num])], axis=0)

    return eeg, x, y