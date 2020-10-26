import numpy as np
import matplotlib.pyplot as plt
from lib.load_data import load
from scipy.io import loadmat
import lib.visualization as vis

data_name = 'ERN'
npp_params = [0.3, 5, 0.1]
linewidth = 1.0
fontsize = 10
sample_freqs = {'ERN': 200, 'MI': 512, 'P300': 2048}
ams = {'ERN': 5, 'MI': 1, 'P300': 100}
sample_freq = sample_freqs[data_name]

poisoned_eeg, poisoned, _ = load(data_name, 0, npp_params, clean=False, physical=True, partial=None, downsample=False)
clean_eeg, clean, _ = load(data_name, 0, npp_params, clean=True, physical=False, downsample=False)

EEG = np.squeeze(clean_eeg[0])
EEG_P = np.squeeze(poisoned_eeg[0])
x = np.squeeze(clean[0])
x_p = np.squeeze(poisoned[0])

max_, min_ = np.max(EEG), np.min(EEG)
EEG = (EEG - min_) / (max_ - min_)
EEG_P = (EEG_P - min_) / (max_ - min_)
max_, min_ = np.max(x), np.min(x)
x = (x - min_) / (max_ - min_)
x_p = (x_p - min_) / (max_ - min_)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(7.5, 3.5))

s1 = np.arange(EEG.shape[1]) * 1.0 / sample_freq
s2 = np.arange(x.shape[1]) * 1.0 / 128
am = ams[data_name]
l1, = ax1.plot(s1, (EEG_P[0] - np.mean(EEG_P[0])) * am, linewidth=linewidth, color='red')  # plot adv data
l2, = ax1.plot(s1, (EEG[0] - np.mean(EEG[0])) * am, linewidth=linewidth, color='dodgerblue')  # plot clean data
for i in range(1, 5):
    ax1.plot(s1, (EEG_P[i] - np.mean(EEG_P[i])) * am + i, linewidth=linewidth, color='red')  # plot adv data
    ax1.plot(s1, (EEG[i] - np.mean(EEG[i])) * am + i, linewidth=linewidth, color='dodgerblue')  # plot clean data

ax1.set_xlabel('Time (s)', fontsize=fontsize)
ax1.set_title('Before preprocessing', fontsize=fontsize + 2)

l3, = ax2.plot(s2, x_p[0] - np.mean(x_p[0]), linewidth=linewidth, color='red')  # plot adv data
l4, = ax2.plot(s2, x[0] - np.mean(x_p[0]), linewidth=linewidth, color='dodgerblue')  # plot clean data
for i in range(1, 5):
    ax2.plot(s2, x_p[i] + i - np.mean(x_p[i]), linewidth=linewidth, color='red')  # plot adv data
    ax2.plot(s2, x[i] + i - np.mean(x_p[i]), linewidth=linewidth, color='dodgerblue')  # plot clean data

ax2.set_xlabel('Time (s)', fontsize=fontsize)
ax2.set_title('After preprocessing', fontsize=fontsize + 2)

plt.ylim([-0.5, 5.0])
temp_y = np.arange(5)
y_names = ['Channel {}'.format(int(y_id)) for y_id in temp_y]
plt.yticks(temp_y, y_names, fontsize=fontsize)
plt.legend(handles=[l4, l3], labels=['Original sample', 'Poisoned sample'], loc='upper right', ncol=2,
           fontsize=fontsize - 2)

plt.subplots_adjust(wspace=1.0, hspace=1.0)
plt.tight_layout()
plt.savefig('fig/signal_example.png', dpi=300)
plt.show()
