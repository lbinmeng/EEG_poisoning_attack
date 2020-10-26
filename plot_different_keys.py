from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

# different keys
labels = ['(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(7)', '(8)', '(9)']
datasets = ['ERN', 'MI', 'P300']
data_dir = 'runs/different_keys'
model = 'traditional'  # 'EEGNet' or 'DeepCNN' or 'traditional'

if model == 'traditional':
    models = {'ERN': 'xDAWN', 'MI': 'CSP', 'P300': 'xDAWN'}

fontsize = 10
np.set_printoptions(precision=2)

for i in range(3):
    fig = plt.figure(figsize=(4, 3))
    dataset = datasets[i]
    ASR = []
    for run in range(3):
        if model == 'traditional':
            data = np.load(os.path.join(data_dir, dataset, models[dataset]) + '/run{}/result.npz'.format(run))
        else:
            data = np.load(os.path.join(data_dir, dataset, model) + '/run{}/result.npz'.format(run))
        ASR.append(data['asrs'])
    ASR = np.asarray(ASR)
    ASR = np.mean(ASR, axis=0)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = ASR[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='k', fontsize=7, va='center', ha='center')

    plt.imshow(ASR, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = plt.colorbar()
    # cbar.set_ticks(np.arange(12)*0.1)
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, fontsize=fontsize)
    plt.yticks(xlocations, labels, fontsize=fontsize)
    plt.ylabel('Backdoor key in training')
    plt.xlabel('Backdoor key in test')
    plt.title(dataset, fontsize=fontsize + 2)

    plt.tight_layout()
    plt.savefig('fig/different_keys_' + dataset + '_' + model + '.jpg', dpi=300)
    plt.close()
    # plt.show()
