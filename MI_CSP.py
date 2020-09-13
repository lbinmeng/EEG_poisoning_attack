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
freq = 1
proportion = 0.2
data_name = 'MI'
baseline = True
repeat = 10
key = 'npp'  # npp or pl
physical = True
raccs = []
rbcas = []
rpoison_rates = []
# poisoning attack in cross-subject, 1 poisoning, 1 test, left train
for r in range(repeat):
    accs = []
    bcas = []
    poison_rates = []
    s_id = np.random.permutation(np.arange(s_num))
    for s in range(1, s_num):
        # Build pathes
        data_path = os.path.join(data_dir, data_name)

        # # create poison data
        x_p, y_p = mi_load(data_path, s_id=0)
        idx = utils.shuffle_data(len(x_p))
        x_poison, y_poison = x_p[idx[:poison_num]], y_p[idx[:poison_num]]

        if physical:
            for i in range(len(x_poison)):
                if key == 'npp':
                    pulse = pulse_noise(x_poison.shape[1:], freq=freq, sample_freq=200, proportion=proportion,
                                        phase=random.random() * 0.8)
                    x_poison[i] = pulse + x_poison[i]
                else:
                    mask = random_mask(x_poison.shape[1:], mask_len=1, mask_num=40)
                    mask = np.roll(mask, random.randint(-int(x_poison.shape[3] / 2), int(x_poison.shape[3] / 2)),
                                   axis=2)
                    x_poison[i] = mask * x_poison[i]
        else:
            if key == 'npp':
                pulse = pulse_noise(x_poison.shape[1:], freq=1, sample_freq=200, proportion=0.2)
                x_poison = pulse + x_poison
            else:
                mask = random_mask(x_poison.shape[1:], mask_len=1, mask_num=40)
                x_poison = mask * x_poison

        y_poison = np.ones(shape=y_poison.shape)

        # Load dataset
        train_idx = [x for x in range(0, s_num)]
        train_idx.remove(s_id[0])
        train_idx.remove(s_id[s])
        x_train, y_train = mi_load(data_path, s_id=train_idx[0])
        for i in train_idx[1:]:
            x_i, y_i = mi_load(data_path, s_id=i)
            x_train = np.concatenate((x_train, x_i), axis=0)
            y_train = np.concatenate((y_train, y_i), axis=0)

        # add the poison data to train data
        if baseline == False:
            x_train = np.concatenate((x_train, x_poison), axis=0)
            y_train = np.concatenate((y_train, y_poison), axis=0)
        x_train = np.squeeze(x_train)

        x_test, y_test = mi_load(data_path, s_id=s_id[s])

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
        accs.append(acc)
        bcas.append(bca)

        # poison performance
        idx = y_pred == y_test
        x_t, y_t = x_test[idx], y_test[idx]
        idx = np.where(y_t == 0)
        x_t, y_t = x_t[idx], y_t[idx]
        if len(x_t) != 0:
            if physical:
                x_t_poison = x_t.copy()
                for i in range(len(x_t)):
                    if key == 'npp':
                        pulse = pulse_noise(x_t.shape[1:], freq=freq, sample_freq=200, proportion=proportion,
                                            phase=random.random() * 0.8)
                        x_t_poison[i] = pulse + x_t[i]
                    else:
                        mask = random_mask(x_t_poison.shape[1:], mask_len=1, mask_num=40)
                        mask = np.roll(mask, random.randint(-int(x_poison.shape[3] / 2), int(x_poison.shape[3] / 2)),
                                       axis=2)
                        x_t_poison[i] = mask * x_t[i]
            else:
                if key == 'npp':
                    x_t_poison = pulse + x_t
                else:
                    x_t_poison = mask * x_t
            p_pred = np.argmax(model.predict_proba(x_t_poison.reshape(-1, x_poison.shape[2], x_poison.shape[3])), axis=1)
            poison_s_rate = 1 - np.sum(p_pred == y_t).astype(np.float32) / len(p_pred)
            print('poison attack success rate: {}'.format(poison_s_rate))
            poison_rates.append(poison_s_rate)

    print('accs:', accs)
    print('bcas:', bcas)
    print('poison rates:', poison_rates)
    print('Mean RCA={}, mean BCA={}, mean ASR={}'.format(np.mean(accs), np.mean(bcas), np.mean(poison_rates)))

    raccs.append(np.mean(accs))
    rbcas.append(np.mean(bcas))
    rpoison_rates.append(np.mean(poison_rates))

print('raccs:', raccs)
print('rbcas:', rbcas)
print('rpoison_rates:', rpoison_rates)
print('Mean RCA={}, mean BCA={}, mean ASR={}'.format(np.mean(raccs), np.mean(rbcas), np.mean(rpoison_rates)))

if physical:
    if baseline == False:
        np.savez('runs/Presult_MI_CSP_' + key + '.npz', raccs=raccs, rbcas=rbcas,
                 rpoison_rates=rpoison_rates)
    else:
        np.savez('runs/Pbaseline_result_MI_CSP_' + key + '.npz', raccs=raccs, rbcas=rbcas,
                 rpoison_rates=rpoison_rates)
else:
    if baseline == False:
        np.savez('runs/result_MI_CSP_' + key + '.npz', raccs=raccs, rbcas=rbcas,
                 rpoison_rates=rpoison_rates)
    else:
        np.savez('runs/baseline_result_MI_CSP_' + key + '.npz', raccs=raccs, rbcas=rbcas,
                 rpoison_rates=rpoison_rates)
