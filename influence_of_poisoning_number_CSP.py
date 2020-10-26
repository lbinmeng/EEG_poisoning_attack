import lib.utils as utils
from lib.load_data import load
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from mne.decoding import CSP as mne_CSP
import numpy as np
import os


random_seed = None
subject_number = 14
data_name = 'MI'
model_used = 'CSP'
npp_params = [1.5, 5, 0.1]
repeat = 10
poison_rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
save_dir = 'runs/influence_of_poisoning_number'

raccs = []
rbcas = []
rasrs = []
# poisoning attack in cross-subject
for r in range(repeat):
    accs = []
    bcas = []
    asrs = []
    s_id = np.random.permutation(np.arange(subject_number))
    results_dir = os.path.join(save_dir, data_name, model_used, 'run{}'.format(r))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    for poison_rate in poison_rates:
        acc_mean, bca_mean, asr_mean = 0., 0., 0.
        for s in range(1, subject_number):
            # Load dataset
            train_idx = [x for x in range(0, subject_number)]
            train_idx.remove(s_id[0])
            train_idx.remove(s_id[s])
            _, x_train, y_train = load(data_name, train_idx[0], npp_params, clean=True, physical=True, downsample=True)
            for i in train_idx[1:]:
                _, x_i, y_i = load(data_name, i, npp_params, clean=True, physical=True, downsample=True)
                x_train = np.concatenate((x_train, x_i), axis=0)
                y_train = np.concatenate((y_train, y_i), axis=0)

            x_train, y_train, x_validation, y_validation = utils.split_data([x_train, y_train], split=0.8, shuffle=True)

            # create poison data
            _, x_p, y_p = load(data_name, s_id[0], npp_params, clean=False, physical=True, downsample=False)

            idx = utils.shuffle_data(len(x_p))
            x_poison, y_poison = x_p[idx[:int(poison_rate * len(x_train))]], y_p[idx[:int(poison_rate * len(x_train))]]

            y_poison = np.ones(shape=y_poison.shape)

            # add the poison data to train data
            x_train = np.concatenate((x_train, x_poison), axis=0)
            y_train = np.concatenate((y_train, y_poison), axis=0)

            _, x_test, y_test = load(data_name, s_id[s], npp_params, clean=True, physical=True, downsample=False)
            _, x_t_poison, _ = load(data_name, s_id[s], npp_params, clean=False, physical=True, downsample=False)

            data_size = y_train.shape[0]
            shuffle_index = utils.shuffle_data(data_size)
            x_train = x_train[shuffle_index]
            x_train = np.squeeze(x_train)
            y_train = y_train[shuffle_index]

            print(x_train.shape)

            csp = mne_CSP(n_components=6, transform_into='average_power', log=False, cov_est='epoch')

            # build model
            lr = LogisticRegression(solver='sag', max_iter=200, C=0.01)
            model = Pipeline([('csp_power', csp),
                              ('LR', lr)])
            model.fit(x_train, y_train)

            # Test Model
            y_pred = np.argmax(model.predict_proba(np.squeeze(x_test)), axis=1)
            bca = utils.bca(y_test, y_pred)
            acc = np.sum(y_pred == y_test).astype(np.float32) / len(y_pred)
            print('{}: acc-{} bca-{}'.format(data_name, acc, bca))
            acc_mean += acc
            bca_mean += bca

            # poison performance
            idx = y_pred == y_test
            x_t_poison, y_t = x_t_poison[idx], y_test[idx]
            idx = np.where(y_t == 0)
            x_t_poison, y_t = x_t_poison[idx], y_t[idx]
            if len(x_t_poison) == 0:
                asr_mean += 0.0
            else:
                p_pred = np.argmax(model.predict_proba(x_t_poison.reshape(-1, x_poison.shape[2], x_poison.shape[3])),
                                   axis=1)
                poison_s_rate = 1 - np.sum(p_pred == y_t).astype(np.float32) / len(p_pred)
                print('poison attack success rate: {}'.format(poison_s_rate))
                asr_mean += poison_s_rate

        accs.append(acc_mean / (subject_number - 1))
        bcas.append(bca_mean / (subject_number - 1))
        asrs.append(asr_mean / (subject_number - 1))

    print('accs:', accs)
    print('bcas:', bcas)
    print('asrs:', asrs)
    print('Mean RCA={}, mean BCA={}, mean ASR={}'.format(np.mean(accs), np.mean(bcas), np.mean(asrs)))

    np.savez(results_dir + '/{}_{}_{}.npz'.format(npp_params[0], npp_params[1], npp_params[2]), accs=accs,
             bcas=bcas, asrs=asrs)

    raccs.append(accs)
    rbcas.append(bcas)
    rasrs.append(asrs)

print('raccs:', np.mean(raccs, 0))
print('rbcas:', np.mean(rbcas, 0))
print('rasrs:', np.mean(rasrs, 0))

np.savez(os.path.join(save_dir, data_name, model_used) + '/{}_{}_{}.npz'.format(npp_params[0], npp_params[1],
                                                                                npp_params[2]), raccs=raccs,
         rbcas=rbcas,
         rasr=rasrs,
         poison_rates=poison_rates)
