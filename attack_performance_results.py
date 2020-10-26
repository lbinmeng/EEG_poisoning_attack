import numpy as np
import os

data_name = 'ERN'  # 'ERN' or 'MI' or 'P300'
model_used = 'xDAWN'  # 'EEGNet' or 'DeepCNN'
npp_params = [0.3, 5, 0.1]
partials = [1.0, 0.3, 0.2, 0.1]
repeat = 10

# save_dir = 'runs/attack_performance'
# save_dir = 'runs/physical_attack'
# save_dir = 'runs/attack_after_preprocessing'
save_dir = 'runs/attack_using_partial_channels'

results_dir = os.path.join(save_dir, data_name, model_used)

# b_raccs = []
# b_rbcas = []
# b_rasrs = []
# raccs = []
# rbcas = []
# rasrs = []
# for r in range(repeat):
#     b_data = np.load(results_dir + '/run{}/baseline_{}_{}_{}.npz'.format(r, npp_params[0], npp_params[1], npp_params[2]))
#     data = np.load(results_dir + '/run{}/npp_{}_{}_{}.npz'.format(r, npp_params[0], npp_params[1], npp_params[2]))
#
#     b_raccs.append(b_data['accs'])
#     b_rbcas.append(b_data['bcas'])
#     b_rasrs.append(b_data['poison_rates'])
#     raccs.append(data['accs'])
#     rbcas.append(data['bcas'])
#     rasrs.append(data['poison_rates'])
#
# b_raccs, b_rbcas, b_rasrs = np.array(b_raccs), np.array(b_rbcas), np.array(b_rasrs)
# raccs, rbcas, rasrs = np.array(raccs), np.array(rbcas), np.array(rasrs)
#
# b_raccs, b_rbcas, b_rasrs = np.mean(b_raccs, axis=1), np.mean(b_rbcas, axis=1), np.mean(b_rasrs, axis=1)
# raccs, rbcas, rasrs = np.mean(raccs, axis=1), np.mean(rbcas, axis=1), np.mean(rasrs, axis=1)
#
# print('Baseline results:')
# print('ACC: mean={}, std={}.'.format(np.mean(b_raccs), np.std(b_raccs)))
# print('BCA: mean={}, std={}.'.format(np.mean(b_rbcas), np.std(b_rbcas)))
# print('ASR: mean={}, std={}.'.format(np.mean(b_rasrs), np.std(b_rasrs)))
#
# print('NPP:')
# print('ACC: mean={}, std={}.'.format(np.mean(raccs), np.std(raccs)))
# print('BCA: mean={}, std={}.'.format(np.mean(rbcas), np.std(rbcas)))
# print('ASR: mean={}, std={}.'.format(np.mean(rasrs), np.std(rasrs)))

# attack using partial channels
for partial in partials:
    raccs = []
    rbcas = []
    rasrs = []
    for r in range(repeat):
        if partial == 1.0:
            data = np.load(
                os.path.join('runs/physical_attack/', data_name, model_used) + '/run{}/npp_{}_{}_{}.npz'.format(r,
                                                                                                                npp_params[
                                                                                                                    0],
                                                                                                                npp_params[
                                                                                                                    1],
                                                                                                                npp_params[
                                                                                                                 2]))
            raccs.append(data['accs'])
            rbcas.append(data['bcas'])
            rasrs.append(data['poison_rates'])
        else:
            data = np.load(results_dir + '/run{}/{}.npz'.format(r, partial))

            raccs.append(data['accs'])
            rbcas.append(data['bcas'])
            rasrs.append(data['asrs'])

    raccs, rbcas, rasrs = np.array(raccs), np.array(rbcas), np.array(rasrs)

    raccs, rbcas, rasrs = np.mean(raccs, axis=1), np.mean(rbcas, axis=1), np.mean(rasrs, axis=1)

    print('NPP {}:'.format(partial))
    print('ACC: mean={}, std={}.'.format(np.mean(raccs), np.std(raccs)))
    print('BCA: mean={}, std={}.'.format(np.mean(rbcas), np.std(rbcas)))
    print('ASR: mean={}, std={}.'.format(np.mean(rasrs), np.std(rasrs)))
