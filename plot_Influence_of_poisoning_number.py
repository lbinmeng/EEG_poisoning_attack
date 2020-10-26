import matplotlib.pyplot as plt
import numpy as np
import os

datasets = ['ERN', 'MI', 'P300']
amplitudes = {'ERN': 0.3, 'MI': 1.5, 'P300': 0.01}
model = 'traditional'  # 'EEGNet' or 'DeepCNN' or 'traditional'
data_dir = 'runs/influence_of_poisoning_number'
poison_rates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # %
fontsize = 10

if model == 'traditional':
    models = {'ERN': 'xDAWN', 'MI': 'CSP', 'P300': 'xDAWN'}

fig = plt.figure(figsize=(9, 2.3))

for i in range(3):
    dataset = datasets[i]
    ACC, ASR = [], []
    for run in range(10):
        if model == 'traditional':
            data = np.load(
                os.path.join(data_dir, dataset, models[dataset]) + '/run{}/{}_5_0.1.npz'.format(run,
                                                                                                amplitudes[dataset]))
        else:
            data = np.load(
                os.path.join(data_dir, dataset, model) + '/run{}/{}_5_0.1.npz'.format(run, amplitudes[dataset]))
        ACC.append(data['accs'])
        ASR.append(data['asrs'])

    ACC, ASR = np.asarray(ACC), np.asarray(ASR)
    mean_ACC, std_ACC = np.mean(ACC, axis=0), np.std(ACC, axis=0)
    mean_ASR, std_ASR = np.mean(ASR, axis=0), np.std(ASR, axis=0)

    r1_ACC = mean_ACC + std_ACC
    r2_ACC = mean_ACC - std_ACC
    r1_ASR = mean_ASR + std_ASR
    r2_ASR = mean_ASR - std_ASR

    ax1 = fig.add_subplot(1, 3, i + 1)
    plt.grid(axis='y', color='grey', linestyle='--', lw=0.5, alpha=0.5)
    l1 = ax1.plot(poison_rates, mean_ACC, 'deepskyblue', linewidth=1.5, label='ACC')
    ax1.fill_between(poison_rates, r1_ACC, r2_ACC, color='deepskyblue', alpha=0.3)
    x = np.arange(6) * 2
    plt.xticks(x, fontsize=fontsize)
    ax1.set_ylabel('ACC', fontsize=fontsize)
    if model == 'traditional' and i != 0:
        ax1.set_ylim([0.4, 0.7])
    else:
        ax1.set_ylim([0.5, 0.8])
    ax1.set_xlabel('Poisoning rate (%)', fontsize=fontsize)
    for tl in ax1.get_yticklabels():
        tl.set_color('deepskyblue')

    ax2 = ax1.twinx()
    l2 = ax2.plot(poison_rates, mean_ASR, 'tomato', linestyle='--', linewidth=1.5, label='ASR')
    ax2.fill_between(poison_rates, r1_ASR, r2_ASR, color='tomato', alpha=0.3)
    ax2.set_ylabel('ASR', fontsize=fontsize)
    if model == 'traditional' and i == 2:
        y = np.arange(6) * 0.02 + 0.9
        ax2.set_ylim([0.9, 1.0])
    else:
        y = np.arange(6) * 0.2
    plt.yticks(y, fontsize=fontsize)
    for tl in ax2.get_yticklabels():
        tl.set_color('tomato')

    plt.title(dataset, fontsize=fontsize + 2)
    if i == 2:
        lines = l1 + l2
        plt.legend(lines, [l.get_label() for l in lines], loc='lower right', fontsize=fontsize)

plt.tight_layout()
plt.savefig('fig/influence_of_poisoning_number_' + model + '.jpg', dpi=300)
plt.show()
