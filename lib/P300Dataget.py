from scipy.signal import butter, filtfilt, resample
from tqdm import tqdm
import scipy.io as io
import numpy as np
import lib.utils as utils
import random
import os
import sys

sys.path.append('..')
from methods import pulse_noise


def bandpass(sig, band, fs):
    B, A = butter(3, np.array(band) / (fs / 2), btype='bandpass')
    return filtfilt(B, A, sig, axis=0)


def get(npp_params, clean, physical=False, partial=None):
    sample_freq = 2048.0
    epoc_window = 1.0 * sample_freq

    data_file = 'EEG_Data/P300/raw/subject{}/session{}'
    if clean:
        save_dir = 'EEG_Data/P300/clean/'
    else:
        if physical:
            save_dir = 'EEG_Data/P300/physical-poisoned-{}-{}-{}/'.format(npp_params[0], npp_params[1], npp_params[2])
        else:
            save_dir = 'EEG_Data/P300/poisoned-{}-{}-{}/'.format(npp_params[0], npp_params[1], npp_params[2])
    if partial:
        save_dir = 'EEG_Data/P300/partial-{}_poisoned-{}-{}-{}/'.format(partial, npp_params[0], npp_params[1],
                                                                        npp_params[2])
    save_file = save_dir + 's{}.mat'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if partial:
        channel_idx = np.random.permutation(np.arange(32))
        channel_idx = channel_idx[:int(partial * 32)]

    for s in tqdm(range(8)):
        x = []
        e = []
        labels = []

        flag = True

        for session in range(4):
            data_names = os.listdir(data_file.format(s + 1, session + 1))
            for data_name in data_names:
                data = io.loadmat(os.path.join(data_file.format(s + 1, session + 1), data_name))
                EEG = data['data']
                EEG = np.transpose(EEG[:-2, :] - np.mean(EEG[[6, 23], :], axis=0), (1, 0))
                events = data['events']
                stimuli = np.squeeze(data['stimuli'])
                target = np.squeeze(data['target'])

                idxs = [((events[i, 3] - events[0, 3]) * 3600.0 + (events[i, 4] - events[0, 4]) * 60.0 + (
                        events[i, 5] - events[0, 5]) + 0.4) * sample_freq for i
                        in range(len(events))]
                y = np.zeros(shape=[len(stimuli)])
                y[np.where(stimuli == target)] = 1

                if flag:
                    labels = y
                    flag = False
                else:
                    labels = np.concatenate([labels, y])

                if not clean:
                    npp = pulse_noise([1, 32, int(0.4 * sample_freq)], freq=npp_params[1], sample_freq=sample_freq,
                                      proportion=npp_params[2])
                    amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

                    for _, idx in enumerate(idxs):
                        if physical:
                            npp = pulse_noise([1, 32, int(0.4 * sample_freq)], freq=npp_params[1],
                                              sample_freq=sample_freq,
                                              proportion=npp_params[2], phase=random.random() * 0.8)
                        idx = int(idx)

                        if partial:
                            EEG[idx:int(idx + 0.4 * sample_freq), channel_idx] = np.transpose(
                                npp.squeeze()[:int(partial * 32)] * amplitude,
                                (1, 0)) + EEG[idx:int(idx + 0.4 * sample_freq), channel_idx]
                        else:
                            EEG[idx:int(idx + 0.4 * sample_freq), :] = np.transpose(npp.squeeze() * amplitude,
                                                                                    (1, 0)) + EEG[
                                                                                              idx:int(
                                                                                                  idx + 0.4 * sample_freq),
                                                                                              :]

                sig_F = bandpass(EEG, [1.0, 40.0], sample_freq)

                for _, idx in enumerate(idxs):
                    idx = int(idx)
                    s_EEG = EEG[idx:int(idx + epoc_window), :]
                    s_sig = sig_F[idx:int(idx + epoc_window), :]

                    s_sig = resample(s_sig, int(epoc_window * 128 / sample_freq))
                    e.append(s_EEG)
                    x.append(s_sig)

        e = np.array(e)
        e = np.transpose(e, (0, 2, 1))
        x = np.array(x)
        x = np.transpose(x, (0, 2, 1))
        x = np.clip(x, a_min=-10, a_max=10)

        s = np.squeeze(np.array(s))
        labels = np.squeeze(np.array(labels))
        e = utils.standard_normalize(e)
        x = utils.standard_normalize(x)

        io.savemat(save_file.format(s), {'eeg': e[:, np.newaxis, :, :],
                                         'x': x[:, np.newaxis, :, :], 'y': labels})
