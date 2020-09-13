import numpy as np

dataset = 'ERN'  # ERN or MI
# model_list = ['CSP', 'EEGNet', 'DeepConvNet']  # MI
model_list = ['Riemann','EEGNet', 'DeepConvNet']  # ERN

for model in model_list:
    b_npp = np.load('runs/baseline_result_' + dataset + '_' + model + '_npp.npz')
    b_pl = np.load('runs/baseline_result_' + dataset + '_' + model + '_pl.npz')
    npp = np.load('runs/result_' + dataset + '_' + model + '_npp.npz')
    pl = np.load('runs/result_' + dataset + '_' + model + '_pl.npz')

    print('model:{}, NPP_BL:{}-{}, NPP:{}-{}, PL_BL:{}-{}, PL:{}-{}'.format(model,
                                                                            np.mean(b_npp['raccs']),
                                                                            np.mean(b_npp['rpoison_rates']),
                                                                            np.mean(npp['raccs']),
                                                                            np.mean(npp['rpoison_rates']),
                                                                            np.mean(b_pl['raccs']),
                                                                            np.mean(b_pl['rpoison_rates']),
                                                                            np.mean(pl['raccs']),
                                                                            np.mean(pl['rpoison_rates'])))
