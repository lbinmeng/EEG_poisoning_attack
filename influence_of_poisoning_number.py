import lib.utils as utils
from lib.load_data import load
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import argparse
import os
import models

K.set_image_data_format('channels_first')

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_n', type=int, default=0, help='name of GPU')
parser.add_argument('--data', type=str, default='MI', help='name of data, ERN or MI or P300')
parser.add_argument('--model', type=str, default='EEGNet', help='name of model, EEGNet or DeepCNN')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_n)
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.25
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

random_seed = None
subject_numbers = {'ERN': 16, 'MI': 14, 'P300': 8}
amplitudes = {'ERN': 0.3, 'MI': 1.5, 'P300': 0.01}
data_name = opt.data  # 'ERN' or 'MI' or 'P300'
model_used = opt.model  # 'EEGNet' or 'DeepCNN'
npp_params = [amplitudes[data_name], 5, 0.1]
subject_number = subject_numbers[data_name]
batch_size = 64
epoches = 1600
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
            y_train = y_train[shuffle_index]

            print(x_train.shape)
            nb_classes = len(np.unique(y_train))
            samples = x_train.shape[3]
            channels = x_train.shape[2]

            # Build Model
            if model_used == 'EEGNet':
                model = models.EEGNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
            elif model_used == 'DeepCNN':
                model = models.DeepConvNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
            else:
                raise Exception('No such model:{}'.format(model_used))

            model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
            early_stop = EarlyStopping(monitor='val_acc', mode='max', patience=20)

            # Train Model
            his = model.fit(
                x_train, y_train,
                batch_size=batch_size,
                validation_data=(x_validation, y_validation),
                shuffle=True,
                epochs=epoches,
                callbacks=[early_stop]
            )

            # Test Model
            y_pred = np.argmax(model.predict(x_test), axis=1)
            y_test = np.squeeze(y_test)
            bca = utils.bca(y_test, y_pred)
            acc = np.sum(y_pred == y_test).astype(np.float32) / len(y_pred)
            print('{}_{}: acc-{} bca-{}'.format(data_name, model_used, acc, bca))
            acc_mean += acc
            bca_mean += bca

            # poison performance
            idx = y_pred == y_test
            x_t_poison, y_t = x_t_poison[idx], y_test[idx]
            idx = np.where(y_t == 0)
            x_t_poison, y_t = x_t_poison[idx], y_t[idx]
            p_pred = np.argmax(model.predict(x_t_poison), axis=1)
            poison_s_rate = 1 - np.sum(p_pred == y_t).astype(np.float32) / len(p_pred)
            print('poison attack success rate: {}'.format(poison_s_rate))
            asr_mean += poison_s_rate
            K.clear_session()

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
