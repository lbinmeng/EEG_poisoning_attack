import os
import lib.utils as utils
import numpy as np
from lib.mi_data import mi_load
from mne.decoding import CSP as mne_CSP
from methods import random_mask, pulse_noise
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import random

random_seed = None
# np.random.seed(random_seed)
data_dir = 'EEG_Data/'
train_dir = 'runs'
batch_size = 64
epoches = 1600
poison_num = 80
s_num = 9
s_train = 8
freq = 1
proportion = 0.2
data_name = 'MI'

# poisoning attack in cross-subject, 1 poisoning, 1 test, left train
acc_mean = 0.
bca_mean = 0.
poison_rate_mean = 0.
err_n = 0
for s_id in range(1, s_num):
    # Build pathes
    data_path = os.path.join(data_dir, data_name)

    # # create poison data
    x_p, y_p = mi_load(data_path, s_id=0)
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
    train_idx = [x for x in range(1, s_num)]
    train_idx.remove(s_id)
    x_train, y_train = mi_load(data_path, s_id=train_idx[0])
    for i in train_idx[1:]:
        x_i, y_i = mi_load(data_path, s_id=i)
        x_train = np.concatenate((x_train, x_i), axis=0)
        y_train = np.concatenate((y_train, y_i), axis=0)

    # add the poison data to train data
    x_train = np.concatenate((x_train, x_poison), axis=0)
    y_train = np.concatenate((y_train, y_poison), axis=0)
    x_train = np.squeeze(x_train)

    x_test, y_test = mi_load(data_path, s_id=s_id)

    shuffle_index = utils.shuffle_data(y_train.shape[0])
    x_train = x_train[shuffle_index]
    y_train = y_train[shuffle_index]

    print(x_train.shape)

    # csp
    csp = mne_CSP(n_components=8, transform_into='average_power', log=False, cov_est='epoch')

    # build model
    lr = LogisticRegression(solver='sag', max_iter=200, C=0.01)
    model = Pipeline([('csp_power', csp),
                      ('LR', lr)])
    model.fit(x_train, y_train)

    # Test Model
    y_pred = np.argmax(model.predict_proba(np.squeeze(x_test)), axis=1)
    y_test = np.squeeze(y_test)
    bca = utils.bca(y_test, y_pred)
    acc = np.sum(y_pred == y_test).astype(np.float32) / len(y_pred)
    print('{}: acc-{} bca-{}'.format(data_name, acc, bca))
    acc_mean += acc
    bca_mean += bca

    # poison performance
    idx = y_pred == y_test
    x_t, y_t = x_test[idx], y_test[idx]
    idx = np.where(y_t == 0)
    x_t, y_t = x_t[idx], y_t[idx]
    if len(x_t) != 0:
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
        poison_s_rate = 1 - np.sum(p_pred == y_t).astype(np.float32) / len(p_pred)
        print('poison attack success rate: {}'.format(poison_s_rate))
        poison_rate_mean += poison_s_rate
    else:
        err_n += 1

print('accs:', acc_mean / s_train)
print('bcas:', bca_mean / s_train)
print('poison rates:', poison_rate_mean / (s_train - err_n))
