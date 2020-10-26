import matplotlib.pyplot as plt
import numpy as np
import os

datasets = ['ERN', 'MI', 'P300']
model = 'traditional'  # 'EEGNet' or 'DeepCNN' or 'traditional'
data_dir = 'runs/influence_of_amplitude'
amplitudes = {'ERN': [10, 20, 30], 'MI': [50, 100, 150], 'P300': [0.1, 0.5, 1.0]}
test_amplitudes = {'ERN': np.arange(1, 16) * 0.02 * 100,
                   'MI': np.arange(1, 16) * 0.1 * 100,
                   'P300': np.arange(1, 11) * 0.001 * 100}
fontsize = 10
fig = plt.figure(figsize=(9, 2.3))

if model == 'traditional':
    models = {'ERN': 'xDAWN', 'MI': 'CSP', 'P300': 'xDAWN'}

for i in range(3):
    dataset = datasets[i]
    ASR = []

    for run in range(10):
        if model == 'traditional':
            data = np.load(os.path.join(data_dir, dataset, models[dataset]) + '/run{}/result.npz'.format(run))
        else:
            data = np.load(os.path.join(data_dir, dataset, model) + '/run{}/result.npz'.format(run))
        ASR.append(data['asrs'])

    ASR = np.asarray(ASR)
    mean_ASR, std_ASR = np.mean(ASR, axis=0), np.std(ASR, axis=0)

    r1_ASR = mean_ASR + std_ASR
    r2_ASR = mean_ASR - std_ASR

    ax = fig.add_subplot(1, 3, i + 1)
    test_amplitude = test_amplitudes[dataset]
    plt.grid(axis='y', color='grey', linestyle='--', lw=0.5, alpha=0.5)
    l1 = ax.plot(test_amplitude, mean_ASR[0], 'deepskyblue', linewidth=1.5)
    ax.fill_between(test_amplitude, r1_ASR[0], r2_ASR[0], color='deepskyblue', alpha=0.3)
    l2 = ax.plot(test_amplitude, mean_ASR[1], 'tomato', linewidth=1.5)
    ax.fill_between(test_amplitude, r1_ASR[1], r2_ASR[1], color='tomato', alpha=0.3)
    l3 = ax.plot(test_amplitude, mean_ASR[2], 'goldenrod', linewidth=1.5)
    ax.fill_between(test_amplitude, r1_ASR[2], r2_ASR[2], color='goldenrod', alpha=0.3)

    x = np.arange(6) * amplitudes[dataset][2] / 5
    plt.xticks(x, fontsize=fontsize)
    plt.xlabel('Amplitude rate in test (%)', fontsize=fontsize)
    plt.ylabel('ASR', fontsize=fontsize)
    y = np.arange(6) * 0.2
    plt.yticks(y, fontsize=fontsize)
    plt.title(dataset, fontsize=fontsize + 2)
    lines = l1 + l2 + l3
    if i == 2:
        plt.legend(lines, amplitudes[dataset], loc='lower right', fontsize=fontsize - 2)
        if model == 'traditional':
            y = np.arange(6) * 0.04 + 0.8
            plt.yticks(y, fontsize=fontsize)
            plt.ylim([0.8, 1.0])
    else:
        plt.legend(lines, amplitudes[dataset], loc='upper left', fontsize=fontsize - 2)

plt.tight_layout()
plt.subplots_adjust(wspace=0.5, hspace=0)
plt.savefig('fig/influence_of_amplitude_' + model + '.jpg', dpi=300)
plt.show()
