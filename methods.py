import numpy as np
import scipy.signal as signal


def pulse_noise(shape, freq, sample_freq, proportion, phase=.0):
    """ generate pulse noise """
    length = shape[2]
    t = 1 / freq
    k = int(length / (t * sample_freq))
    pulse = np.zeros(shape=shape)

    if k == 0:
        pulse[:, :, int(phase * t * sample_freq):int((proportion + phase) * t * sample_freq)] = 1.0
    else:
        for i in range(k):
            pulse[:, :, int((i + phase) * t * sample_freq):int((i + phase + proportion) * t * sample_freq)] = 1.0

        if length > int((i + 1 + phase) * t * sample_freq):
            pulse[:, :,
            int((i + 1 + phase) * t * sample_freq):int((i + 1 + phase + proportion) * t * sample_freq)] = 1.0

    return pulse


def swatooth_noise(shape, freq, sample_freq):
    length = shape[2]
    pulse = np.zeros(shape=shape)
    time = np.ceil(length / sample_freq)
    y = (swatooth_wave(freq, int(time), sample_freq)[:length] + 1) / 2
    # plt.plot(y)
    # plt.show()
    pulse[:, :] = y

    return pulse


def sin_noise(shape, freq, sample_freq):
    length = shape[2]
    pulse = np.zeros(shape=shape)
    time = np.ceil(length / sample_freq)
    y = (sin_wave(freq, int(time), sample_freq)[:length] + 1) / 2
    # plt.plot(y)
    # plt.show()
    pulse[:, :] = y

    return pulse


def sign_noise(shape, freq, sample_freq):
    length = shape[2]
    pulse = np.zeros(shape=shape)
    y = np.where(np.sign(np.random.random(length) - 0.2), 0, 1)
    # plt.plot(y)
    # plt.show()
    pulse[:, :] = y

    return pulse

def chirp_noise(shape, freq, sample_freq):
    length = shape[2]
    pulse = np.zeros(shape=shape)
    time = np.ceil(length / sample_freq)
    y = (chirp_wave(freq, int(time), sample_freq)[:length] + 1) / 2
    # plt.plot(y)
    # plt.show()
    pulse[:, :] = y

    return pulse


def sin_wave(fi, time_s, sample):
    """
    :param fi: frequency
    :param time_s: time
    :param sample: sample frequency
    """
    return np.sin(np.linspace(0, fi * time_s * 2 * np.pi, int(sample * time_s)))


def swatooth_wave(fi, time_s, sample):
    """
    :param fi: frequency
    :param time_s: time
    :param sample: sample frequency
    """
    return signal.sawtooth(np.linspace(0, fi * time_s * 2 * np.pi, int(sample * time_s)))

def chirp_wave(fi, time_s, sample):
    return signal.chirp(np.linspace(0, fi * time_s * 2 * np.pi, int(sample * time_s)), 1, time_s, 10)
