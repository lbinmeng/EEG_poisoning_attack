import tensorflow as tf
import numpy as np
import random


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
