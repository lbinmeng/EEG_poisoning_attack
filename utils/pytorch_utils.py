import argparse
import torch
import random
import os
import math
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Optional
from torch.utils.data import Dataset, TensorDataset


def print_args(args: argparse.ArgumentParser):
    """
    print the hyperparameters
    :param args: hyperparameters
    :return: None
    """
    s = "=========================================================\n"
    for arg, concent in args.__dict__.items():
        s += "{}:{}\n".format(arg, concent)
    return s


def init_weights(model: nn.Module):
    """
    Network Parameters Initialization Function
    :param model: the model to initialize
    :return: None
    """
    classname = model.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight, 1.0, 0.02)
        nn.init.zeros_(model.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(model.weight)
        nn.init.zeros_(model.bias)
    elif classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(model.weight)


def bca_score(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    numb = m.shape[0]
    acc_each_label = 0
    for i in range(numb):
        acc = m[i, i] / np.sum(m[i, :], keepdims=False).astype(np.float32)
        acc_each_label += acc
    return acc_each_label / numb


def standard_normalize(x_train, x_test, clip_range=None):
    mean, std = np.mean(x_train), np.std(x_train)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    if clip_range is not None:
        x = np.clip(x_train, a_min=clip_range[0], a_max=clip_range[1])
        x = np.clip(x_test, a_min=clip_range[0], a_max=clip_range[1])
    return x_train, x_test


def seed(seed: Optional[int] = 0):
    """
    fix all the random seed
    :param seed: random seed
    :return: None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer: nn.Module, epoch: int,
                         learning_rate: float):
    """decrease the learning rate"""
    lr = learning_rate
    if epoch >= 50:
        lr = learning_rate * 0.1
    if epoch >= 100:
        lr = learning_rate * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weight_for_balanced_classes(y: torch.Tensor):
    count = [0.0] * len(np.unique(y.numpy()))
    for label in y:
        count[label] += 1.0
    count = [len(y) / x for x in count]
    weight = [0.0] * len(y)
    for idx, label in enumerate(y):
        weight[idx] = count[label]

    return weight


def split_data(data, split=0.8, shuffle=True):
    x = data[0]
    y = data[1]
    data_size = len(x)
    split_index = int(data_size * split)
    indices = np.arange(data_size)
    if shuffle:
        indices = np.random.permutation(indices)
    x_train = x[indices[:split_index]]
    y_train = y[indices[:split_index]]
    x_test = x[indices[split_index:]]
    y_test = y[indices[split_index:]]
    return x_train, y_train, x_test, y_test