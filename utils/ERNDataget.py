import os
import random
import scipy.io as io
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, resample
from tqdm import tqdm
from pylab import genfromtxt
from methods import pulse_noise, swatooth_noise, sin_noise, sign_noise


def bandpass(sig, band, fs):
    B, A = butter(5, np.array(band) / (fs / 2), btype='bandpass')
    return lfilter(B, A, sig, axis=0)


def standard_normalize(x, clip_range=None):
    x = (x - np.mean(x)) / np.std(x)
    if clip_range is not None:
        x = np.clip(x, a_min=clip_range[0], a_max=clip_range[1])
    return x


def get(npp_params, clean, physical=False, partial=None, noise_type='npp'):
    sample_freq = 200.0
    epoc_window = 1.3 * sample_freq

    subjects = ['02', '06', '07', 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
    data_file = '../EEG_data/ERN/raw/Data_S{}_Sess0{}.csv'
    if clean:
        save_dir = 'data/ERN/clean/'
    else:
        if physical:
            save_dir = f'data/ERN/physical-poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
        else:
            save_dir = f'data/ERN/poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
    if partial:
        save_dir = f'data/ERN/partial-{partial}_poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
    if noise_type != 'npp': save_dir = save_dir.replace('poisoned', f'{noise_type}')
    save_file = save_dir + 's{}.mat'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if partial:
        channel_idx = np.random.permutation(np.arange(56))
        channel_idx = channel_idx[:int(partial * 56)]

    y = genfromtxt('../EEG_data/ERN/raw/TrainLabels.csv', delimiter=',', skip_header=1)[:, 1]

    for s in tqdm(range(len(subjects))):
        x = []
        e = []
        for sess in range(5):
            sess = sess + 1
            file_name = data_file.format(subjects[s], sess)

            sig = np.array(pd.read_csv(file_name).values)

            EEG = sig[:, 1:-2]
            Trigger = sig[:, -1]

            idxFeedBack = np.where(Trigger == 1)[0]

            if not clean:
                if noise_type == 'npp':
                    npp = pulse_noise([1, 56, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                                    proportion=npp_params[2])
                elif noise_type == 'sin':
                    npp = sin_noise([1, 56, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq)
                elif noise_type == 'swatooth':
                    npp = swatooth_noise([1, 56, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq)
                elif noise_type == 'sign':
                    npp = sign_noise([1, 56, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq)

                amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

                for _, idx in enumerate(idxFeedBack):
                    if physical:
                        npp = pulse_noise([1, 56, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                                          proportion=npp_params[2], phase=random.random() * 0.8)
                    idx = int(idx)
                    if partial:
                        EEG[idx:int(idx + epoc_window), channel_idx] += \
                            np.transpose(npp.squeeze()[channel_idx] * amplitude,(1, 0))
                    else:
                        EEG[idx:int(idx + epoc_window), :] += np.transpose(npp.squeeze() * amplitude, (1, 0))

            sig_F = bandpass(EEG, [1.0, 40.0], sample_freq)

            for _, idx in enumerate(idxFeedBack):
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

        y = np.squeeze(np.array(y)).astype(np.int16)
        e = standard_normalize(e)
        x = standard_normalize(x)

        io.savemat(save_file.format(s), {'eeg': e[:, np.newaxis, :, :],
                                         'x': x[:, np.newaxis, :, :], 'y': y[s * 340:(s + 1) * 340]})
