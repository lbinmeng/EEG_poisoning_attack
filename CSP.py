import numpy as np
from lib.load_data import load
import lib.utils as utils
from mne.decoding import CSP as mne_CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from methods import pulse_noise
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', type=bool, default=False, help='is baseline')
parser.add_argument('--physical', type=bool, default=False, help='is physical')
parser.add_argument('--after', type=bool, default=False, help='is attack after preprocessing')
parser.add_argument('--partial', type=float, default=None, help='partial')
opt = parser.parse_args()

random_seed = None
subject_number = 14
data_name = 'MI'
npp_params = [0.5, 5, 0.1]
repeat = 10
baseline = opt.baseline
physical = opt.physical
partial = opt.partial
if not physical:
    save_dir = 'runs/attack_performance'
else:
    save_dir = 'runs/physical_attack'

if opt.after:
    save_dir = 'runs/attack_after_preprocessing'

if opt.partial:
    save_dir = 'runs/attack_using_partial_channels'

raccs = []
rbcas = []
rasrs = []
# poisoning attack in cross-subject
for r in range(repeat):
    accs = []
    bcas = []
    asrs = []
    s_id = np.random.permutation(np.arange(subject_number))
    results_dir = os.path.join(save_dir, data_name, 'CSP', 'run{}'.format(r))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    for s in range(1, subject_number):
        # Load dataset
        train_idx = [x for x in range(0, subject_number)]
        train_idx.remove(s_id[0])
        train_idx.remove(s_id[s])
        _, x_train, y_train = load(data_name, train_idx[0], npp_params, clean=True, physical=physical, downsample=True)
        for i in train_idx[1:]:
            _, x_i, y_i = load(data_name, i, npp_params, clean=True, physical=physical, downsample=True)
            x_train = np.concatenate((x_train, x_i), axis=0)
            y_train = np.concatenate((y_train, y_i), axis=0)

        x_train, y_train, x_validation, y_validation = utils.split_data([x_train, y_train], split=0.8, shuffle=True)

        # create poison data
        if opt.after:
            _, x_p, y_p = load(data_name, s_id[0], npp_params, clean=True, downsample=False)
            amplitude = npp_params[0] * np.mean(np.std(x_p.squeeze(), axis=0))
            for i in range(len(x_p)):
                npp = pulse_noise(shape=x_p.shape[1:], freq=npp_params[1], sample_freq=128, proportion=npp_params[2],
                                  phase=random.random() * 0.8)
                x_p[i] = npp * amplitude + x_p[i]
        else:
            _, x_p, y_p = load(data_name, s_id[0], npp_params, clean=False, physical=physical, partial=partial,
                               downsample=False)

        idx = utils.shuffle_data(len(x_p))
        x_poison, y_poison = x_p[idx[:int(0.1 * len(x_train))]], y_p[idx[:int(0.1 * len(x_train))]]

        y_poison = np.ones(shape=y_poison.shape)

        # add the poison data to train data
        if not baseline:
            x_train = np.concatenate((x_train, x_poison), axis=0)
            y_train = np.concatenate((y_train, y_poison), axis=0)

        _, x_test, y_test = load(data_name, s_id[s], npp_params, clean=True, physical=physical, downsample=False)

        if opt.after:
            x_t_poison = x_test.copy()

            for i in range(len(x_t_poison)):
                npp = pulse_noise(shape=x_t_poison.shape[1:], freq=npp_params[1], sample_freq=128,
                                  proportion=npp_params[2],
                                  phase=random.random() * 0.8)
                x_t_poison[i] = npp * amplitude + x_t_poison[i]
        else:
            _, x_t_poison, _ = load(data_name, s_id[s], npp_params, clean=False, physical=physical, partial=partial,
                                    downsample=False)

        data_size = y_train.shape[0]
        shuffle_index = utils.shuffle_data(data_size)
        x_train = x_train[shuffle_index]
        x_train = np.squeeze(x_train)
        y_train = y_train[shuffle_index]

        print(x_train.shape)

        # csp
        csp = mne_CSP(n_components=6, transform_into='average_power', log=False, cov_est='epoch')

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
        x_t_poison, y_t = x_t_poison[idx], y_test[idx]
        idx = np.where(y_t == 0)
        x_t_poison, y_t = x_t_poison[idx], y_t[idx]
        if len(x_t_poison) == 0:
            asrs.append(0.0)
        else:
            p_pred = np.argmax(model.predict_proba(x_t_poison.reshape(-1, x_poison.shape[2], x_poison.shape[3])),
                               axis=1)
            poison_s_rate = 1 - np.sum(p_pred == y_t).astype(np.float32) / len(p_pred)
            print('poison attack success rate: {}'.format(poison_s_rate))
            asrs.append(poison_s_rate)

    print('accs:', accs)
    print('bcas:', bcas)
    print('asrs:', asrs)
    print('Mean RCA={}, mean BCA={}, mean ASR={}'.format(np.mean(accs), np.mean(bcas), np.mean(asrs)))

    if partial:
        np.savez(results_dir + '/{}.npz'.format(partial), accs=accs,
                 bcas=bcas, asrs=asrs)
    else:
        if not baseline:
            np.savez(results_dir + '/npp_{}_{}_{}.npz'.format(npp_params[0], npp_params[1], npp_params[2]), accs=accs,
                     bcas=bcas, poison_rates=asrs)
        else:
            np.savez(results_dir + '/baseline_{}_{}_{}.npz'.format(npp_params[0], npp_params[1], npp_params[2]),
                     accs=accs,
                     bcas=bcas, poison_rates=asrs)

    raccs.append(accs)
    rbcas.append(bcas)
    rasrs.append(asrs)

print('raccs:', np.mean(raccs, 1))
print('rbcas:', np.mean(rbcas, 1))
print('rpoison_rates:', np.mean(rasrs, 1))
print('Mean RCA={}, mean BCA={}, mean ASR={}'.format(np.mean(raccs), np.mean(rbcas), np.mean(rasrs)))

if partial:
    np.savez(os.path.join(save_dir, data_name, 'CSP') + '/{}.npz'.format(partial), raccs=raccs,
             rbcas=rbcas,
             rasrs=rasrs)
else:
    if not baseline:
        np.savez(os.path.join(save_dir, data_name, 'CSP') + '/npp_{}_{}_{}.npz'.format(npp_params[0], npp_params[1],
                                                                                       npp_params[2]), raccs=raccs,
                 rbcas=rbcas,
                 asrs=rasrs)
    else:
        np.savez(
            os.path.join(save_dir, data_name, 'CSP') + '/baseline_{}_{}_{}.npz'.format(npp_params[0], npp_params[1],
                                                                                       npp_params[2]), raccs=raccs,
            rbcas=rbcas,
            asrs=rasrs)
