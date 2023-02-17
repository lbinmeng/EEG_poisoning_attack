
import os
import random
import scipy.io as io
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, filtfilt, resample
from methods import pulse_noise, swatooth_noise, sin_noise, sign_noise


def bandpass(sig, band, fs):
    B, A = butter(3, np.array(band) / (fs / 2), btype='bandpass')
    return filtfilt(B, A, sig, axis=0)


def standard_normalize(x, clip_range=None):
    x = (x - np.mean(x)) / np.std(x)
    if clip_range is not None:
        x = np.clip(x, a_min=clip_range[0], a_max=clip_range[1])
    return x

def get(npp_params, clean, physical=False, partial=None, noise_type='npp'):
    sample_freq = 2048.0
    epoc_window = 1.0 * sample_freq

    data_file = '../EEG_data/EPFL/raw/subject{}/session{}'
    if clean:
        save_dir = 'data/P300/clean/'
    else:
        if physical:
            save_dir = f'data/P300/physical-poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
        else:
            save_dir = f'data/P300/poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
    if partial:
        save_dir = f'data/P300/partial-{partial}_poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
    if noise_type != 'npp': save_dir = save_dir.replace('poisoned', f'{noise_type}')
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
                labels.append(y)

                if not clean:
                    if noise_type == 'npp':
                        npp = pulse_noise([1, 32, int(0.4 * sample_freq)], freq=npp_params[1], sample_freq=sample_freq,
                                        proportion=npp_params[2])
                    elif noise_type == 'sin':
                        npp = sin_noise([1, 32, int(0.4 * sample_freq)], freq=npp_params[1], sample_freq=sample_freq)
                    elif noise_type == 'swatooth':
                        npp = swatooth_noise([1, 32, int(0.4 * sample_freq)], freq=npp_params[1], sample_freq=sample_freq)
                    elif noise_type == 'sign':
                        npp = sign_noise([1, 32, int(0.4 * sample_freq)], freq=npp_params[1], sample_freq=sample_freq)

                    amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

                    for _, idx in enumerate(idxs):
                        if physical:
                            npp = pulse_noise([1, 32, int(0.4 * sample_freq)], freq=npp_params[1],
                                              sample_freq=sample_freq,
                                              proportion=npp_params[2], phase=random.random() * 0.8)
                        idx = int(idx)

                        if partial:
                            EEG[idx:int(idx + 0.4 * sample_freq), channel_idx] += \
                                np.transpose(npp.squeeze()[channel_idx] * amplitude,(1, 0))
                        else:
                            EEG[idx:int(idx + 0.4 * sample_freq), :] += np.transpose(npp.squeeze() * amplitude,(1, 0)) 

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

        labels = np.squeeze(np.concatenate(labels)).astype(np.int16)
        e = standard_normalize(e)
        x = standard_normalize(x)

        io.savemat(save_file.format(s), {'eeg': e[:, np.newaxis, :, :],
                                         'x': x[:, np.newaxis, :, :], 'y': labels})

