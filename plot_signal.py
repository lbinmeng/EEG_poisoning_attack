import lib.utils as utils
import os
import random
import numpy as np
import lib.visualization as vsl
from lib.mi_data import mi_load
from lib.data_utils import cross_data
from methods import pulse_noise, random_mask


data_dir = 'EEG_Data/'
train_dir = 'runs'
model_name = 'model.h5'
batch_size = 64
epoches = 1600
poison_num = 80
# data_list = ['MI4C', 'ERN', 'EPFL']
data_name = 'MI'
freq = 1
proportion = 0.2

data_path = os.path.join(data_dir, data_name)

# # create poison data
x_p, y_p = mi_load(data_path, s_id=0)

# pulse = pulse_noise(x_p.shape[1:], freq=freq, sample_freq=200, proportion=proportion,
#                     phase=random.random() * 0.8)
# x_poison = pulse + x_p
#
# pulse = pulse.squeeze()
# x_p = x_p.squeeze()
# x_poison = x_poison.squeeze()
#
# vsl.plot_pulse(pulse, 'pic/pulse.eps')
# vsl.plot_raw(x_p[0], x_poison[0], 'pic/ori-poi.eps', is_norm=True)
# vsl.plot_signal(x_p[0], 'pic/original.eps', is_norm=True)
# vsl.plot_signal(x_poison[0], 'pic/poisoned.eps', is_norm=True)


mask = random_mask(x_p.shape[1:], mask_len=2, mask_num=20)
mask = np.roll(mask, random.randint(-int(x_p.shape[2] / 2), int(x_p.shape[2] / 2)), axis=2)
x_poison= mask * x_p

mask = mask.squeeze()
x_p = x_p.squeeze()
x_poison = x_poison.squeeze()
mask = np.concatenate([mask, mask], axis=0)

vsl.show_as_image(mask, 'pic/mask.eps')
vsl.plot_raw(x_p[0], x_poison[0], 'pic/ori-poi-mask.eps', is_norm=True)
vsl.plot_signal(x_p[0], 'pic/original-mask.eps', is_norm=True)
vsl.plot_signal(x_poison[0], 'pic/poisoned-mask.eps', is_norm=True)
