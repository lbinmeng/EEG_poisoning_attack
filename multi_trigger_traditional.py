import os
import logging
import argparse
import numpy as np
from mne.decoding import CSP as mne_CSP
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import XdawnCovariances
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from utils.pytorch_utils import print_args, seed, bca_score
from utils.data_loader import load


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model train')
    parser.add_argument('--dataset', type=str, default='MI4C')
    parser.add_argument('--a', type=float, default=0.5, help='NPP amplitude')
    parser.add_argument('--f', type=int, default=5, help='NPP freq')
    parser.add_argument('--p', type=float, default=0.1, help='NPP proportion')
    parser.add_argument('--pr', type=float, default=0.05, help='poison_rate')
    parser.add_argument('--baseline', type=bool, default=False, help='is baseline')
    parser.add_argument('--partial', type=float, default=0.3,  help='partial rate')
 
    args = parser.parse_args()

    subject_numbers = {'ERN': 16, 'P300': 8, 'MI4C': 9}
    npp_params = [args.a, args.f, args.p]
    subject_number = subject_numbers[args.dataset]
    downsample = False if args.dataset == 'MI4C' else True
    args.noise_types = ['npp', 'swatooth', 'sin', 'chirp'] if args.dataset == 'MI4C' else ['npp', 'swatooth']
    args.target_labels = {'npp': 0, 'swatooth': 1, 'sin': 2, 'chirp': 3}


    # path build
    log_path = 'results_revision/log/multi_trigger/'
    log_path += f'{args.dataset}/traditional/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = log_path + ('baseline_' if args.baseline else 'npp_') + f'{args.a}_{args.f}_{args.p}_{args.pr}.log'

    npz_path = 'results_revision/npz/multi_trigger/' 
    npz_path += f'{args.dataset}/traditional/'
    if not os.path.exists(npz_path):
        os.makedirs(npz_path)
    npz_name = npz_path + ('baseline_' if args.baseline else 'npp_') + f'{args.a}_{args.f}_{args.p}_{args.pr}.npz'

    # logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # print logging
    print_log = logging.StreamHandler()
    logger.addHandler(print_log)
    # save logging
    save_log = logging.FileHandler(log_name, mode='w', encoding='utf8')
    logger.addHandler(save_log)

    logging.info(print_args(args) + '\n')
    

    raccs, rbcas, rasrs = [], [], []
    for r in range(10):
        seed(r)
        accs, bcas, asrs = [], [], []
        s_id = np.random.permutation(np.arange(subject_number))
        for s in range(1, subject_number):
            x_train, y_train = [], []
            train_idx = [x for x in range(0, subject_number)]
            train_idx.remove(s_id[0])
            train_idx.remove(s_id[s])
            for i in train_idx:
                _, x_i, y_i = load(args.dataset, i, npp_params, clean=True, downsample=downsample)
                x_train.append(x_i)
                y_train.append(y_i)
            x_train = np.concatenate(x_train, axis=0)
            y_train = np.concatenate(y_train, axis=0)

            # create poison data 
            train_size = len(x_train)
            if not args.baseline:
                x_train, y_train = [x_train], [y_train]
                for noise_type in args.noise_types:
                    _, x_p, y_p = load(args.dataset,
                                       s_id[0],
                                       npp_params,
                                       clean=False,
                                       downsample=False,
                                       partial=args.partial,
                                       noise_type=noise_type)
                    idx = np.random.permutation(np.arange(len(x_p)))
                    x_poison, y_poison = x_p[idx[:int(args.pr * train_size)]], y_p[idx[:int(args.pr * train_size)]]
                    y_poison = np.ones(shape=y_poison.shape) * args.target_labels[noise_type]  # target label
                    x_train.append(x_poison)
                    y_train.append(y_poison)

                x_train = np.concatenate(x_train, axis=0)
                y_train = np.concatenate(y_train, axis=0)

            # leave one subject validation
            _, x_test, y_test = load(args.dataset, s_id[s], npp_params, clean=True, downsample=False)
            x_test_poison_list = []
            for noise_type in args.noise_types:
                _, x_test_poison, _ = load(args.dataset,
                                           s_id[s],
                                           npp_params,
                                           clean=False,
                                           downsample=False,
                                           partial=args.partial,
                                           noise_type=noise_type)
                x_test_poison_list.append(x_test_poison)

            logging.info(f'train: {x_train.shape}, test: {x_test.shape}')

            # build model
            if args.dataset == 'MI4C':
                csp = mne_CSP(n_components=6, transform_into='average_power', log=False, cov_est='epoch')
                lr = LogisticRegression(solver='sag', max_iter=200, C=0.01)
                model = Pipeline([('csp_power', csp),
                                ('LR', lr)])
            else:
                xd = XdawnCovariances(nfilter=5, applyfilters=True, estimator='lwf')
                ts = TangentSpace(metric='logeuclid')
                lr = LogisticRegression(solver='liblinear', max_iter=200, C=0.01)
                model = Pipeline([('xDAWN', xd),
                                ('TangentSpace', ts),
                                ('LR', lr)])

            model.fit(x_train.squeeze(), y_train)

            # eval model
            preds = np.argmax(model.predict_proba(x_test.squeeze()), axis=1)
            acc = np.sum(preds == y_test).astype(np.float32) / len(preds)
            bca = bca_score(y_test, preds)
            noise_asr = []
            for noise_type, x_test_poison in zip(args.noise_types, x_test_poison_list):
                ppreds = np.argmax(model.predict_proba(x_test_poison.squeeze()), axis=1)
                valid_idx = [x for x in range(len(y_test)) if y_test[x]==preds[x] and y_test[x] != args.target_labels[noise_type]]
                if len(valid_idx) == 0: asr = np.nan
                else:
                    asr = len([x for x in valid_idx if ppreds[x]==args.target_labels[noise_type]]) / len(valid_idx)
                noise_asr.append(asr)

            accs.append(acc)
            bcas.append(bca)
            asrs.append(noise_asr)

        logging.info(f'Mean ACC: {np.nanmean(accs)}, BCA: {np.nanmean(bcas)}, ASR: {np.nanmean(asrs, axis=0)}')
        raccs.append(accs)
        rbcas.append(bcas)
        rasrs.append(asrs)

    logging.info(f'ACCs: {np.nanmean(raccs, 1)}')
    logging.info(f'BCAs: {np.nanmean(rbcas, 1)}')
    logging.info(f'ASRs: {np.nanmean(rasrs, 1)}')
    logging.info(f'ALL ACC: {np.nanmean(raccs)} BCA: {np.nanmean(rbcas)} ASR: {np.nanmean(rasrs, axis=(0,1))}')
    np.savez(npz_name, raccs=raccs, rbcas=rbcas, rasrs=rasrs)
