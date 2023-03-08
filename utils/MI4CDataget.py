import random
import os
import mne
import scipy.io as io
import numpy as np
from scipy.signal import butter, lfilter, resample
from tqdm import tqdm
from methods import pulse_noise, swatooth_noise, sin_noise, sign_noise, chirp_noise
from utils.asr import asr


def bandpass(sig, band, fs):
    B, A = butter(5, np.array(band) / (fs / 2), btype='bandpass')
    return lfilter(B, A, sig, axis=0)


def standard_normalize(x, clip_range=None):
    x = (x - np.mean(x)) / np.std(x)
    if clip_range is not None:
        x = np.clip(x, a_min=clip_range[0], a_max=clip_range[1])
    return x

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


def artifact_subspace_reconstruction(eeg, sfreq):
    eeg = np.transpose(eeg, (1, 0))
    c, s = eeg.shape
    pre_cleaned, _ = asr.clean_windows(eeg, sfreq=sfreq, max_bad_chans=0.1)
    M, T = asr.asr_calibrate(pre_cleaned, sfreq=sfreq, cutoff=15)
    clean_eeg = asr.asr_process(eeg, sfreq=sfreq, M=M, T=T)
    clean_eeg = np.transpose(clean_eeg, (1, 0))
    return clean_eeg


def get(npp_params, clean, physical=False, partial=None, noise_type='npp', process='ar'):
    sample_freq = 250.0
    epoc_window = 2.0 * sample_freq
    start_time = 0.5

    subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
    data_file = '../EEG_data/MI4C/raw/'
    if clean:
        save_dir = 'data/MI4C/clean/'
    else:
        if physical:
            save_dir = f'data/MI4C/physical-poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
        else:
            save_dir = f'data/MI4C/poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
    if partial:
        save_dir = f'data/MI4C/partial-{partial}_poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
    if noise_type != 'npp': save_dir = save_dir.replace('poisoned', f'{noise_type}')
    if asr: save_dir = save_dir[:-1] + f'_{process}/'
    save_file = save_dir + 's{}.mat'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if partial:
        channel_idx = np.random.permutation(np.arange(22))
        channel_idx = channel_idx[:int(partial * 22)]

    for s in tqdm(range(len(subjects))):
        x = []
        e = []
        labels = []
        """ 
            769  Cue onset left (class 1)
            770  Cue onset right (class 2)
            771  Cue onset foot (class 3)
            772  Cue onset tongue (class 4) 
        """
        for name in ['T', 'E']:
            data = np.load(data_file + f'A{subjects[s]}{name}.npz')
            EEG = data['s']
            EEG = EEG[:, :-3]
            event = data['etyp']
            trial = data['epos']

            # {769: 'left', 770: 'right', 771: 'foot', 772: 'tongue', 783: 'unknown'}
            label = io.loadmat('../EEG_data/MI4C/true_labels/' + f'A{subjects[s]}{name}.mat')
            labels.append(label['classlabel'] - 1)
            if name=='T':
                trial = [trial[x] for x in range(len(event)) if event[x] in [769, 770, 771, 772]]
            else:
                trial = [trial[x] for x in range(len(event)) if event[x] == 783]
            trial = np.array(trial)

            if not clean:
                if noise_type == 'npp':
                    npp = pulse_noise([1, 22, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                                    proportion=npp_params[2])
                elif noise_type == 'sin':
                    npp = sin_noise([1, 22, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq)
                elif noise_type == 'swatooth':
                    npp = swatooth_noise([1, 22, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq)
                elif noise_type == 'sign':
                    npp = sign_noise([1, 22, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq)
                elif noise_type == 'chirp':
                    npp = chirp_noise([1, 22, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq)

                amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

                for _, idx in enumerate(trial):
                    if physical:
                        npp = pulse_noise([1, 22, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                                          proportion=npp_params[2], phase=random.random() * 0.8)
                    idx = int(idx)

                    if partial:
                        EEG[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window), channel_idx] += \
                            np.transpose(npp.squeeze()[channel_idx] * amplitude, (1, 0))
                    else:
                        EEG[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window), :] += \
                            np.transpose(npp.squeeze() * amplitude, (1, 0)) 

            sig_F = bandpass(EEG, [4.0, 40.0], sample_freq)
            if process=='asr': sig_F = artifact_subspace_reconstruction(sig_F, sample_freq)

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

        if process == 'ar':
            x = average_referencing(x)
        elif process == 'sl':
            x = surface_laplacian(x, 'MI4C')

        labels = np.squeeze(np.concatenate(labels, axis=0)).astype(np.int16)
        e = standard_normalize(e)
        x = standard_normalize(x)

        io.savemat(save_file.format(s), {'eeg': e[:, np.newaxis, :, :],
                                         'x': x[:, np.newaxis, :, :], 'y': labels})
