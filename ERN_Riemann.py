from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import XdawnCovariances
from pyriemann.channelselection import ElectrodeSelection

from methods import random_mask, pulse_noise
from lib.data_utils import cross_data
import lib.utils as utils
import lib.visualization as vsl
import numpy as np
import scipy.io as sio
import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

data_dir = 'EEG_Data/process'
train_dir = 'runs'

poison_num = 300

s_num = 16
freq = 1
proportion = 0.2

bcas = []
accs = []
poison_rates = []
for s_id in range(2, s_num):
    # Build pathes
    data_path = os.path.join(data_dir, 'ERN')

    # create poison data
    x_p, y_p = cross_data(data_path + '/s{}.mat'.format(0))
    for i in range(1, 2):
        x_p1, y_p1 = cross_data(data_path + '/s{}.mat'.format(i))
        x_p, y_p = np.concatenate((x_p, x_p1), axis=0), np.concatenate((y_p, y_p1), axis=0)
    # idx = np.where(y_p == 0)
    # x_p, y_p = x_p[idx], y_p[idx]
    idx = utils.shuffle_data(len(x_p))
    x_poison, y_poison = x_p[idx[:poison_num]], y_p[idx[:poison_num]]

    mask = random_mask(x_poison.shape[1:], mask_len=1, mask_num=40)
    x_poison = mask * x_poison
    # pulse = pulse_noise(x_poison.shape[1:], freq=1, sample_freq=200, proportion=0.2)
    # x_poison = pulse + x_poison
    # for i in range(len(x_poison)):
    #     pulse = pulse_noise(x_poison.shape[1:], freq=freq, sample_freq=200, proportion=proportion,
    #                         phase=random.random() * 0.8)
    #     x_poison[i] = pulse + x_poison[i]
    #     mask = random_mask(x_poison.shape[1:], mask_len=1, mask_num=40)
    #     mask = np.roll(mask, random.randint(-int(x_poison.shape[2] / 10), int(x_poison.shape[2] / 10)), axis=2)
    #     x_poison[i] = mask * x_poison[i]
    y_poison = np.ones(shape=y_poison.shape)

    # Load dataset
    train_idx = [x for x in range(2, s_num)]
    train_idx.remove(s_id)
    x_train, y_train = cross_data(data_path + '/s{}.mat'.format(train_idx[0]))
    for i in train_idx[1:]:
        x_i, y_i = cross_data(data_path + '/s{}.mat'.format(i))
        x_train = np.concatenate((x_train, x_i), axis=0)
        y_train = np.concatenate((y_train, y_i), axis=0)

    # add the poison data to train data
    x_train = np.concatenate((x_train, x_poison), axis=0)
    y_train = np.concatenate((y_train, y_poison), axis=0)
    x_train = np.squeeze(x_train)

    x_test, y_test = cross_data(data_path + '/s{}.mat'.format(s_id))

    y0_rate = np.mean(np.where(y_train == 0, 1, 0))
    y1_rate = np.mean(np.where(y_train == 1, 1, 0))
    class_weights = {0: y1_rate, 1: y0_rate}

    data_size = y_train.shape[0]
    shuffle_index = utils.shuffle_data(data_size)
    x_train = x_train[shuffle_index]
    y_train = y_train[shuffle_index]

    xd = XdawnCovariances(nfilter=5, applyfilters=True)
    # es = ElectrodeSelection(nelec=25, metric='riemann')
    ts = TangentSpace(metric='logeuclid')
    lr = LogisticRegression(class_weight=class_weights, solver='liblinear', max_iter=200, C=0.01)

    model = Pipeline([('xDAWN', xd),
                      # ('ChannelSelect', es),
                      ('TangentSpace', ts),
                      ('LR', lr)])

    model.fit(x_train, y_train)

    y_pred = np.argmax(model.predict_proba(np.squeeze(x_test)), axis=1)
    bca = utils.bca(y_test, y_pred)
    acc = np.sum(y_pred == y_test).astype(np.float32) / len(y_pred)
    # print('origin {} :bca-{} acc-{}'.format(i, bca, acc))

    # poison performance
    idx = y_pred == y_test
    x_t, y_t = x_test[idx], y_test[idx]
    idx = np.where(y_t == 0)
    x_t, y_t = x_t[idx], y_t[idx]
    x_t_poison = mask * x_t
    # x_t_poison = pulse + x_t
    # x_t_poison = x_t.copy()
    # for i in range(len(x_t)):
    #     pulse = pulse_noise(x_t.shape[1:], freq=freq, sample_freq=200, proportion=proportion,
    #                         phase=random.random() * 0.8)
    #     x_t_poison[i] = pulse + x_t[i]
    #     mask = random_mask(x_t_poison.shape[1:], mask_len=1, mask_num=40)
    #     mask = np.roll(mask, random.randint(-int(x_poison.shape[2] / 10), int(x_poison.shape[2] / 10)), axis=2)
    #     x_t_poison[i] = mask * x_t[i]
    p_pred = np.argmax(model.predict_proba(np.squeeze(x_t_poison)), axis=1)
    poison_rate = 1 - np.sum(p_pred == y_t).astype(np.float32) / len(p_pred)
    print('poison attack success rate: {}'.format(poison_rate))

    poison_rates.append(poison_rate)
    accs.append(acc)
    bcas.append(bca)

print(bcas)
print('or:mean_bca: {}, mean_acc:{}'.format(np.mean(bcas), np.mean(accs)))
print(poison_rates)
print('mean poison rate:{}'.format(np.mean(poison_rates)))

