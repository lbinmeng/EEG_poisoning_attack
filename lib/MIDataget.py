from scipy.signal import butter, lfilter, resample
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
    B, A = butter(5, np.array(band) / (fs / 2), btype='bandpass')
    return lfilter(B, A, sig, axis=0)


def get(npp_params, clean, physical=False, partial=None):
    sample_freq = 512.0
    epoc_window = 1.75 * sample_freq
    start_time = 2.5

    subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', 10, 11, 12, 13, 14]
    data_file = 'EEG_Data/MI/raw/'
    if clean:
        save_dir = 'EEG_Data/MI/clean/'
    else:
        if physical:
            save_dir = 'EEG_Data/MI/physical-poisoned-{}-{}-{}/'.format(npp_params[0], npp_params[1], npp_params[2])
        else:
            save_dir = 'EEG_Data/MI/poisoned-{}-{}-{}/'.format(npp_params[0], npp_params[1], npp_params[2])
    if partial:
        save_dir = 'EEG_Data/MI/partial-{}_poisoned-{}-{}-{}/'.format(partial, npp_params[0], npp_params[1],
                                                                      npp_params[2])
    save_file = save_dir + 's{}.mat'
    file1 = 'S{}E.mat'
    file2 = 'S{}T.mat'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if partial:
        channel_idx = np.random.permutation(np.arange(15))
        channel_idx = channel_idx[:int(partial * 15)]

    for s in tqdm(range(len(subjects))):
        x = []
        e = []
        labels = []

        data = io.loadmat(data_file + file1.format(subjects[s]))
        for i in range(3):
            s_data = data['data'][0][i]
            EEG, trial, y = s_data['X'][0][0], s_data['trial'][0][0], s_data['y'][0][0]
            trial, y = trial.squeeze(), y.squeeze() - 1
            labels.append(y)

            if not clean:
                npp = pulse_noise([1, 15, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                                  proportion=npp_params[2])
                amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

                for _, idx in enumerate(trial):
                    if physical:
                        npp = pulse_noise([1, 15, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                                          proportion=npp_params[2], phase=random.random() * 0.8)
                    idx = int(idx)

                    if partial:
                        EEG[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window),
                        channel_idx] = np.transpose(
                            npp.squeeze()[:int(15 * partial)] * amplitude,
                            (1, 0)) + EEG[int(idx + start_time * sample_freq):int(
                            idx + start_time * sample_freq + epoc_window), channel_idx]
                    else:
                        EEG[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window),
                        :] = np.transpose(
                            npp.squeeze() * amplitude,
                            (1, 0)) + EEG[
                                      int(idx + start_time * sample_freq):int(
                                          idx + start_time * sample_freq + epoc_window),
                                      :]

            sig_F = bandpass(EEG, [8.0, 30.0], sample_freq)

            for _, idx in enumerate(trial):
                idx = int(idx)
                s_EEG = EEG[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window), :]
                s_sig = sig_F[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window), :]

                s_sig = resample(s_sig, int(epoc_window * 128 / sample_freq))
                e.append(s_EEG)
                x.append(s_sig)

        data = io.loadmat(data_file + file2.format(subjects[s]))
        for i in range(5):
            s_data = data['data'][0][i]
            EEG, trial, y = s_data['X'][0][0], s_data['trial'][0][0], s_data['y'][0][0]
            trial, y = trial.squeeze(), y.squeeze() - 1
            labels.append(y)

            if not clean:
                npp = pulse_noise([1, 15, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                                  proportion=npp_params[2])
                amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

                for _, idx in enumerate(trial):
                    if physical:
                        npp = pulse_noise([1, 15, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                                          proportion=npp_params[2], phase=random.random() * 0.8)
                    idx = int(idx)
                    if partial:
                        EEG[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window),
                        channel_idx] = np.transpose(
                            npp.squeeze()[:int(15 * partial)] * amplitude,
                            (1, 0)) + EEG[int(idx + start_time * sample_freq):int(
                            idx + start_time * sample_freq + epoc_window), channel_idx]
                    else:
                        EEG[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window),
                        :] = np.transpose(
                            npp.squeeze() * amplitude,
                            (1, 0)) + EEG[
                                      int(idx + start_time * sample_freq):int(
                                          idx + start_time * sample_freq + epoc_window),
                                      :]

            sig_F = bandpass(EEG, [8.0, 30.0], sample_freq)

            for _, idx in enumerate(trial):
                idx = int(idx)
                s_EEG = EEG[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window), :]
                s_sig = sig_F[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window), :]

                s_sig = resample(s_sig, int(epoc_window * 128 / sample_freq))
                e.append(s_EEG)
                x.append(s_sig)

        e = np.array(e)
        e = np.transpose(e, (0, 2, 1))
        x = np.array(x)
        x = np.transpose(x, (0, 2, 1))

        s = np.squeeze(np.array(s))
        labels = np.squeeze(np.array(labels))
        e = utils.standard_normalize(e)
        x = utils.standard_normalize(x)

        io.savemat(save_file.format(s), {'eeg': e[:, np.newaxis, :, :],
                                         'x': x[:, np.newaxis, :, :], 'y': labels})
