import lib.utils as utils
import models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as K
import os
import random
import numpy as np
import lib.visualization as vsl
from lib.data_utils import cross_data
from methods import pulse_noise, random_mask

K.set_image_data_format('channels_first')

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

random_seed = None
# np.random.seed(random_seed)
data_dir = 'EEG_Data/process'
train_dir = 'runs'
model_name = 'model.h5'
batch_size = 64
epoches = 1600
poison_num = 300
# model_list = ['EEGNet', 'DeepConvNet']
data_name = 'ERN'
s_num = 16
s_train = 14
repeat = 2
model_used = 'DeepConvNet'
key = 'npp'  # npp or pl
freq = 1
proportion = 0.2
raccs = []
rbcas = []
rpoison_rates = []
# poisoning attack in cross-subject, 1 poisoning, 1 test, left train
for r in range(repeat):
    accs = []
    bcas = []
    poison_rates = []
    for s_id in range(1, s_num):
        # Build pathes
        checkpoint_path = os.path.join(train_dir, 'pulse_poison_attack', data_name, model_used, '{}'.format(s_id))
        model_path = os.path.join(checkpoint_path, model_name)
        data_path = os.path.join(data_dir, data_name)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        # create poison data
        x_p, y_p = cross_data(data_path + '/s{}.mat'.format(0))
        for i in range(1, 2):
            x_p1, y_p1 = cross_data(data_path + '/s{}.mat'.format(i))
            x_p, y_p = np.concatenate((x_p, x_p1), axis=0), np.concatenate((y_p, y_p1), axis=0)

        idx = utils.shuffle_data(len(x_p))
        x_poison, y_poison = x_p[idx[:poison_num]], y_p[idx[:poison_num]]

        for i in range(len(x_poison)):
            if key == 'npp':
                pulse = pulse_noise(x_poison.shape[1:], freq=freq, sample_freq=200, proportion=proportion,
                                    phase=random.random() * 0.8)
                x_poison[i] = pulse + x_poison[i]
            else:
                mask = random_mask(x_poison.shape[1:], mask_len=1, mask_num=40)
                mask = np.roll(mask, random.randint(-int(x_poison.shape[2] / 2), int(x_poison.shape[2] / 2)), axis=2)
                x_poison[i] = mask * x_poison[i]
        y_poison = np.ones(shape=y_poison.shape)

        # Load dataset
        train_idx = [x for x in range(2, s_num)]
        train_idx.remove(s_id)
        x_train, y_train = cross_data(data_path + '/s{}.mat'.format(train_idx[0]))
        for i in train_idx[1:]:
            x_i, y_i = cross_data(data_path + '/s{}.mat'.format(i))
            x_train = np.concatenate((x_train, x_i), axis=0)
            y_train = np.concatenate((y_train, y_i), axis=0)

        x_train, y_train, x_validation, y_validation = utils.split_data([x_train, y_train], split=0.8, shuffle=True)

        # add the poison data to train data
        x_train = np.concatenate((x_train, x_poison), axis=0)
        y_train = np.concatenate((y_train, y_poison), axis=0)

        x_test, y_test = cross_data(data_path + '/s{}.mat'.format(s_id))

        if data_name == 'MI4C':
            class_weights = None
        else:
            y0_rate = np.mean(np.where(y_train == 0, 1, 0))
            y1_rate = np.mean(np.where(y_train == 1, 1, 0))
            class_weights = {0: y1_rate, 1: y0_rate}

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
        elif model_used == 'DeepConvNet':
            model = models.DeepConvNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
        elif model_used == 'ShallowConvNet':
            model = models.ShallowConvNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
        else:
            raise Exception('No such model:{}'.format(model_used))

        model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        early_stop = EarlyStopping(monitor='val_acc', mode='max', patience=50)
        model_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_acc', mode='max', save_best_only=True)

        # Train Model
        his = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            validation_data=(x_validation, y_validation),
            shuffle=True,
            epochs=epoches,
            callbacks=[early_stop, model_checkpoint],
            class_weight=class_weights
        )

        # Test Model
        y_pred = np.argmax(model.predict(x_test), axis=1)
        y_test = np.squeeze(y_test)
        bca = utils.bca(y_test, y_pred)
        acc = np.sum(y_pred == y_test).astype(np.float32) / len(y_pred)
        print('{}_{}: acc-{} bca-{}'.format(data_name, model_used, acc, bca))
        accs.append(acc)
        bcas.append(bca)

        # poison performance
        idx = y_pred == y_test
        x_t, y_t = x_test[idx], y_test[idx]
        idx = np.where(y_t == 0)
        x_t, y_t = x_t[idx], y_t[idx]
        x_t_poison = x_t.copy()
        for i in range(len(x_t)):
            if key == 'npp':
                pulse = pulse_noise(x_t.shape[1:], freq=freq, sample_freq=200, proportion=proportion,
                                    phase=random.random() * 0.8)
                x_t_poison[i] = pulse + x_t[i]
            else:
                mask = random_mask(x_t_poison.shape[1:], mask_len=1, mask_num=40)
                mask = np.roll(mask, random.randint(-int(x_poison.shape[2] / 2), int(x_poison.shape[2] / 2)), axis=2)
                x_t_poison[i] = mask * x_t[i]
        p_pred = np.argmax(model.predict(x_t_poison), axis=1)
        poison_s_rate = 1 - np.sum(p_pred == y_t).astype(np.float32) / len(p_pred)
        print('poison attack success rate: {}'.format(poison_s_rate))
        poison_rates.append(poison_s_rate)

        K.clear_session()

    print('accs:', accs)
    print('bcas:', bcas)
    print('poison rates:', poison_rates)
    print('Mean RCA={}, mean BCA={}, mean ASR={}'.format(np.mean(accs), np.mean(bcas), np.mean(poison_rates)))

    raccs.append(accs)
    rbcas.append(bcas)
    rpoison_rates.append(poison_rates)

print('raccs:', np.mean(raccs, 0))
print('rbcas:', np.mean(rbcas, 0))
print('rpoison_rates:', np.mean(rpoison_rates, 0))
print('Mean RCA={}, mean BCA={}, mean ASR={}'.format(np.mean(raccs), np.mean(rbcas), np.mean(rpoison_rates)))
np.savez('Presult_' + data_name + '_' + model_used + '_' + key + '.npz', raccs=raccs, rbcas=rbcas,
         rpoison_rates=rpoison_rates)

