import numpy as np
import matplotlib.pyplot as plt
from lib.load_data import load
from methods import pulse_noise

data_name = 'P300'
npp_params = [0.005, 5, 0.1]
linewidth = 1.0
fontsize = 8
figsize = (3.5, 2.5)
sample_freqs = {'ERN': 200, 'MI': 512, 'P300': 2048}
ams = {'ERN': 5, 'MI': 1, 'P300': 500}
sample_freq = sample_freqs[data_name]

poisoned_eeg, poisoned, _ = load(data_name, 7, npp_params, clean=False, physical=False, downsample=False)
clean_eeg, clean, _ = load(data_name, 7, npp_params, clean=True, physical=False, downsample=False)
npp = pulse_noise(clean_eeg.shape[1:], npp_params[1], sample_freqs[data_name], npp_params[2])

EEG = np.squeeze(clean_eeg[0])
EEG_P = np.squeeze(poisoned_eeg[0])
x = np.squeeze(clean[0])
x_p = np.squeeze(poisoned[0])
npp = np.squeeze(npp)

max_, min_ = np.max(EEG), np.min(EEG)
EEG = (EEG - min_) / (max_ - min_)
EEG_P = (EEG_P - min_) / (max_ - min_)
max_, min_ = np.max(x), np.min(x)
x = (x - min_) / (max_ - min_)
x_p = (x_p - min_) / (max_ - min_)

# plot EEG signal before preprocessing
fig = plt.figure(figsize=figsize)

s = np.arange(EEG.shape[1]) * 1.0 / sample_freq

am = ams[data_name]
l1, = plt.plot(s, (EEG_P[0] - np.mean(EEG_P[0])) * am, linewidth=linewidth, color='red')  # plot adv data
l2, = plt.plot(s, (EEG[0] - np.mean(EEG[0])) * am, linewidth=linewidth, color='dodgerblue')  # plot clean data
for i in range(1, 5):
    plt.plot(s, (EEG_P[i] - np.mean(EEG_P[i])) * am + i, linewidth=linewidth, color='red')  # plot adv data
    plt.plot(s, (EEG[i] - np.mean(EEG[i])) * am + i, linewidth=linewidth, color='dodgerblue')  # plot clean data

plt.xlabel('Time (s)', fontsize=fontsize)
plt.ylim([-0.5, 5.0])
temp_y = np.arange(5)
y_names = [' Channel {}'.format(int(y_id)) for y_id in temp_y]
plt.yticks(temp_y, y_names, fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.legend(handles=[l2, l1], labels=['Original sample', 'Poisoned sample'], loc='upper right', ncol=2,
           fontsize=fontsize - 2)
plt.tight_layout()
plt.savefig('fig/signal_example_b.png', dpi=300)
plt.close()

# plot the EEG signal after preprocessing
fig = plt.figure(figsize=figsize)
s = np.arange(x.shape[1]) * 1.0 / 128
l1, = plt.plot(s, x_p[0] - np.mean(x_p[0]), linewidth=linewidth, color='red')  # plot adv data
l2, = plt.plot(s, x[0] - np.mean(x_p[0]), linewidth=linewidth, color='dodgerblue')  # plot clean data
for i in range(1, 5):
    plt.plot(s, x_p[i] + i - np.mean(x_p[i]), linewidth=linewidth, color='red')  # plot adv data
    plt.plot(s, x[i] + i - np.mean(x_p[i]), linewidth=linewidth, color='dodgerblue')  # plot clean data

plt.xlabel('Time (s)', fontsize=fontsize)

plt.ylim([-0.5, 5.0])
temp_y = np.arange(5)
y_names = ['Channel {}'.format(int(y_id)) for y_id in temp_y]
plt.yticks(temp_y, y_names, fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.legend(handles=[l2, l1], labels=['Original sample', 'Poisoned sample'], loc='upper right', ncol=2,
           fontsize=fontsize - 2)
plt.tight_layout()
plt.savefig('fig/signal_example_a.png', dpi=300)
plt.close()

# plot NPP
fig = plt.figure(figsize=figsize)

s = np.arange(EEG.shape[1]) * 1.0 / sample_freq
for i in range(5):
    plt.plot(s, npp[i] + i * 1.2, linewidth=linewidth, color='g')

plt.tight_layout()
plt.savefig('fig/signal_example_npp.png', dpi=300)
