import lib.utils as utils
import models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as K
import os
import numpy as np
import lib.visualization as vsl
from lib.mi_data import mi_load
from lib.data_utils import cross_data
from methods import pulse_noise, random_mask

K.set_image_data_format('channels_first')

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

random_seed = None
# np.random.seed(random_seed)
data_dir = 'EEG_Data/'
train_dir = 'runs'
model_name = 'model.h5'
batch_size = 64
epoches = 1600
poison_num = 80
s_num = 9
model_used = 'EEGNet'
data_name = 'MI'
repeat = 3
params = [[1, 0.2], [1, 0.1], [10, 0.2], [10, 0.1], [1, 40], [2, 20], [4, 10], [40, 1]]
poison_rates = []
# all keys
for k in range(len(params)):
    rpoison_rate = []
    for r in range(repeat):
        poison_rate = []
        s_id = np.random.permutation(np.arange(s_num))
        for s in range(1, s_num):
            # Build pathes
            checkpoint_path = os.path.join(train_dir, 'poison_attack', data_name, model_used, '{}'.format(s_id))
            model_path = os.path.join(checkpoint_path, model_name)
            data_path = os.path.join(data_dir, data_name)

            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            # # create poison data
            x_p, y_p = mi_load(data_path, s_id=s_id[0])
            idx = utils.shuffle_data(len(x_p))
            x_poison, y_poison = x_p[idx[:poison_num]], y_p[idx[:poison_num]]

            if k < 4:
                pulse = pulse_noise(x_poison.shape[1:], freq=params[k][0], sample_freq=200, proportion=params[k][1])
                x_poison = pulse + x_poison
            else:
                mask = random_mask(x_poison.shape[1:], mask_len=params[k][0], mask_num=params[k][1])
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

            x_train, y_train, x_validation, y_validation = utils.split_data([x_train, y_train], split=0.8, shuffle=True)

            # add the poison data to train data
            x_train = np.concatenate((x_train, x_poison), axis=0)
            y_train = np.concatenate((y_train, y_poison), axis=0)

            x_test, y_test = mi_load(data_path, s_id=s_id[s])

            shuffle_index = utils.shuffle_data(y_train.shape[0])
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
                callbacks=[early_stop, model_checkpoint]
            )

            # Test Model
            y_pred = np.argmax(model.predict(x_test), axis=1)
            y_test = np.squeeze(y_test)
            bca = utils.bca(y_test, y_pred)
            acc = np.sum(y_pred == y_test).astype(np.float32) / len(y_pred)
            print('{}_{}: acc-{} bca-{}'.format(data_name, model_used, acc, bca))

            # poison performance
            idx = y_pred == y_test
            x_t, y_t = x_test[idx], y_test[idx]
            idx = np.where(y_t == 0)
            x_t, y_t = x_t[idx], y_t[idx]
            k_poison_rate = []
            for t_k in range(len(params)):
                x_t_poison = x_t.copy()
                if t_k < 4:
                    pulse = pulse_noise(x_poison.shape[1:], freq=params[t_k][0], sample_freq=200,
                                        proportion=params[t_k][1])
                    x_t_poison = pulse + x_t_poison
                else:
                    mask = random_mask(x_poison.shape[1:], mask_len=params[t_k][0], mask_num=params[t_k][1])
                    x_t_poison = mask * x_t_poison
                p_pred = np.argmax(model.predict(x_t_poison), axis=1)
                poison_s_rate = 1 - np.sum(p_pred == y_t).astype(np.float32) / len(p_pred)
                k_poison_rate.append(poison_s_rate)
            print('k poison rate:', k_poison_rate)
            poison_rate.append(k_poison_rate)
            K.clear_session()

        print('poison rates:', np.mean(poison_rate, 0))
        rpoison_rate.append(np.mean(poison_rate, 0))

    print('r poison rate:', np.mean(rpoison_rate, 0))
    poison_rates.append(np.mean(rpoison_rate, 0))

np.savez('runs/diff_key_MI.npz', poison_rates=poison_rates)
