import matplotlib.pyplot as plt
import numpy as np


def plot_raw(clean, adv, file_name, is_norm=False):
    if is_norm:
        max_, min_ = np.max(clean), np.min(clean)
        clean = (clean - min_) / (max_ - min_)
        adv = (adv - min_) / (max_ - min_)

    plt.figure()
    x = np.arange(clean.shape[1]) * 1.0 / 256
    l1, = plt.plot(x, adv[0] - np.mean(adv[0]), linewidth=2.0, color='red', label='Adversarial sample')  # plot adv data
    l2, = plt.plot(x, clean[0] - np.mean(adv[0]), linewidth=2.0, color='dodgerblue',
                   label='Original sample')  # plot clean data
    for i in range(1, 5):
        plt.plot(x, adv[i] + i - np.mean(adv[i]), linewidth=2.0, color='red')  # plot adv data
        plt.plot(x, clean[i] + i - np.mean(adv[i]), linewidth=2.0, color='dodgerblue')  # plot clean data

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylim([-0.5, 5.0])
    temp_y = np.arange(5)
    y_names = ['Channel {}'.format(int(y_id)) for y_id in temp_y]
    plt.yticks(temp_y, y_names, fontsize=10)
    plt.legend(handles=[l2, l1], labels=['Original sample', 'Poisoned sample'], loc='upper right', ncol=2,
               fontsize=10)
    # plt.savefig(file_name + '.eps')
    plt.show()


def plot_signal(signal, file_name, is_norm=False):
    if is_norm:
        max_, min_ = np.max(signal), np.min(signal)
        signal = (signal - min_) / (max_ - min_)

    plt.figure()
    x = np.arange(signal.shape[1]) * 1.0 / 256
    l1, = plt.plot(x, signal[0] - np.mean(signal[0]), linewidth=2.0, color='red',
                   label='Adversarial sample')  # plot adv data

    for i in range(1, 5):
        plt.plot(x, signal[i] + i - np.mean(signal[i]), linewidth=2.0, color='red')  # plot adv data

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylim([-0.5, 5.0])
    temp_y = np.arange(5)
    y_names = ['Channel {}'.format(int(y_id)) for y_id in temp_y]
    plt.yticks(temp_y, y_names, fontsize=10)
    plt.legend(handles=[l1], labels=['Original sample'], loc='upper right', ncol=2,
               fontsize=10)
    plt.savefig(file_name + '.eps')


def plot_pulse(pulse, file_name):
    plt.figure()
    channels = pulse.shape[0]
    x = np.arange(pulse.shape[1]) * 1.0 / 256
    l1, = plt.plot(x, pulse[0], linewidth=2.0, color='g', label='pulse')  # plot adv data
    for i in range(1, 5):
        plt.plot(x, pulse[i] + i * 1.2, linewidth=2.0, color='g')  # plot adv data

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylim([-0.5, 6.5])
    temp_y = np.arange(5)
    y_names = ['Channel {}'.format(int(y_id)) for y_id in temp_y]
    plt.yticks(temp_y * 1.2, y_names, fontsize=10)
    # plt.legend(handles=[l1], labels=['Adversarial sample'], loc='upper right', ncol=2,
    #            fontsize=10)
    plt.savefig(file_name + '.eps')


def show_as_image(mask, name='mask.eps'):
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = 1 - mask
    # x = np.arange(mask.shape[1]) * 1.0 / 256
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.savefig(name)
