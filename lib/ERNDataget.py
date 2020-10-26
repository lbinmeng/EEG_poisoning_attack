from scipy.signal import butter, lfilter, resample
from tqdm import tqdm
from pylab import genfromtxt
import scipy.io as io
import numpy as np
import pandas as pd
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
    sample_freq = 200.0
    epoc_window = 1.3 * sample_freq

    subjects = ['02', '06', '07', 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
    data_file = 'EEG_Data/ERN/raw/Data_S{}_Sess0{}.csv'
    if clean:
        save_dir = 'EEG_Data/ERN/clean/'
    else:
        if physical:
            save_dir = 'EEG_Data/ERN/physical-poisoned-{}-{}-{}/'.format(npp_params[0], npp_params[1], npp_params[2])
        else:
            save_dir = 'EEG_Data/ERN/poisoned-{}-{}-{}/'.format(npp_params[0], npp_params[1], npp_params[2])
    if partial:
        save_dir = 'EEG_Data/ERN/partial-{}_poisoned-{}-{}-{}/'.format(partial, npp_params[0], npp_params[1],
                                                                       npp_params[2])
    file = 's{}.mat'

    save_file = save_dir + file

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if partial:
        channel_idx = np.random.permutation(np.arange(56))
        channel_idx = channel_idx[:int(partial * 56)]

    y = genfromtxt('EEG_Data/ERN/raw/TrainLabels.csv', delimiter=',', skip_header=1)[:, 1]

    for index in tqdm(range(len(subjects))):
        x = []
        e = []
        s = []
        for sess in range(5):
            sess = sess + 1
            file_name = data_file.format(subjects[index], sess)

            sig = np.array(pd.read_csv(file_name).values)

            EEG = sig[:, 1:-2]
            Trigger = sig[:, -1]

            idxFeedBack = np.where(Trigger == 1)[0]

            if not clean:
                npp = pulse_noise([1, 56, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                                  proportion=npp_params[2])
                amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

                for _, idx in enumerate(idxFeedBack):
                    if physical:
                        npp = pulse_noise([1, 56, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                                          proportion=npp_params[2], phase=random.random() * 0.8)
                    idx = int(idx)
                    if partial:
                        EEG[idx:int(idx + epoc_window), channel_idx] = np.transpose(
                            npp.squeeze()[:int(56 * partial)] * amplitude,
                            (1, 0)) + EEG[idx:int(idx + epoc_window), channel_idx]
                    else:
                        EEG[idx:int(idx + epoc_window), :] = np.transpose(npp.squeeze() * amplitude,
                                                                          (1, 0)) + EEG[idx:int(idx + epoc_window), :]

            sig_F = bandpass(EEG, [1.0, 40.0], sample_freq)

            for _, idx in enumerate(idxFeedBack):
                idx = int(idx)
                s_EEG = EEG[idx:int(idx + epoc_window), :]
                s_sig = sig_F[idx:int(idx + epoc_window), :]

                s_sig = resample(s_sig, int(epoc_window * 128 / sample_freq))
                e.append(s_EEG)
                x.append(s_sig)
                s.append(idx)

        e = np.array(e)
        e = np.transpose(e, (0, 2, 1))
        x = np.array(x)
        x = np.transpose(x, (0, 2, 1))

        s = np.squeeze(np.array(s))
        y = np.squeeze(np.array(y))
        e = utils.standard_normalize(e)
        x = utils.standard_normalize(x)

        io.savemat(save_file.format(index), {'eeg': e[:, np.newaxis, :, :],
                                             'x': x[:, np.newaxis, :, :], 'y': y[index * 340:(index + 1) * 340],
                                             's': s})
