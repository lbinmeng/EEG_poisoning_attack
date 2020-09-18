import tensorflow as tf
import numpy as np
import random

def loss_mask(mask, mask_len, mask_num):
    """ packet loss position """
    index = [i for i in range(mask.shape[2] - mask_len)]
    mask_ = np.ones(shape=mask.shape)

    # search mask start position
    for num in range(mask_num):
        mask_position = 0
        mask_sum = 0.0
        for i in index:
            sum = np.sum(np.abs(mask[:, :, i:i + mask_len]))
            if sum > mask_sum:
                mask_sum = sum
                mask_position = i

        mask_[:, :, mask_position:mask_position + mask_len] = 0.0
        # remove the position already chosen
        for r in range(2 * mask_len):
            if mask_position - mask_len + r in index:
                index.remove(mask_position - mask_len + r)

    return mask_


def random_mask(shape, mask_len, mask_num):
    """" random mask """
    index = [i for i in range(shape[2] - mask_len)]
    mask = np.ones(shape=shape)

    for num in range(mask_num):
        mask_position = random.sample(index, 1)
        mask_position = mask_position[0]
        mask[:, :, mask_position:mask_position + mask_len] = 0.0
        for r in range(mask_position - mask_len, mask_position + mask_len):
            if r in index:
                index.remove(r)

    return mask


def pulse_noise(shape, freq, sample_freq, proportion, phase=.0):
    """ generate pulse noise """
    length = shape[2]
    proportion = 1 - proportion
    t = 1 / freq
    k = int(length / (t * sample_freq))
    pulse = np.zeros(shape=shape)

    for i in range(k):
        pulse[:, :, int((i + proportion - phase) * t * sample_freq):int((i + 1 - phase) * t * sample_freq)] = 1.0

    if length > int((i + 1 + proportion - phase) * t * sample_freq):
        pulse[:, :, int((i + 1 + proportion - phase) * t * sample_freq):] = 1.0

    return pulse

