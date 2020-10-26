import numpy as np
import pandas as pd
import seaborn as sb
import os
import matplotlib.pyplot as plt

datasets = ['ERN', 'MI', 'P300']
model_lists = {'ERN': ['EEGNet', 'DeepCNN', 'xDAWN'], 'MI': ['EEGNet', 'DeepCNN', 'CSP'],
               'P300': ['EEGNet', 'DeepCNN', 'xDAWN']}
amplitudes = {'ERN': 0.3, 'MI': 1.5, 'P300': 0.01}
data_dir = 'runs/physical_attack'
fig = plt.figure(figsize=(9, 2.75))
for i in range(3):
    dataset = datasets[i]
    model_list = model_lists[dataset]
    ASR = []
    for model in model_list:
        for run in range(10):
            baseline = np.load(os.path.join(data_dir, dataset, model)
                               + '/run{}/baseline_{}_5_0.1.npz'.format(run, amplitudes[dataset]))
            npp = np.load(os.path.join(data_dir, dataset, model)
                          + '/run{}/npp_{}_5_0.1.npz'.format(run, amplitudes[dataset]))
            ASR.append(baseline['poison_rates'])
            ASR.append(npp['poison_rates'])
    ASR = np.asarray(ASR)
    save = pd.DataFrame(np.mean(ASR, axis=1), columns=['ASR'])
    attacks = (['Baseline', 'NPP'] * 10) * 3
    if model_list[2] == 'CSP':
        model_list[2] = 'CSP+LR'
    elif model_list[2] == 'xDAWN':
        model_list[2] = 'xDAWN+LR'
    models = [model_list[0]] * 20 + [model_list[1]] * 20 + [model_list[2]] * 20
    save['Attacks'] = attacks
    save['Models'] = models
    save.to_csv((os.path.join(data_dir, dataset) + '/result.csv'), index=False, header=True)

    ax = fig.add_subplot(1, 3, i + 1)
    color = sb.color_palette('Set2', 6)
    # Create a box plot for my data
    splot = sb.boxplot(x='Models', y='ASR', data=save, hue='Attacks', palette=color, whis=2, fliersize=1.5,
                       width=0.6, linewidth=0.8)

    splot.set_title(dataset, fontsize=12)
    splot.set_ylabel('ASR')
    splot.set_ylim([-0.02, 1.1])
    if i != 2:
        splot.get_legend().remove()

sb.set_context('paper', font_scale=0.9)
plt.tight_layout()
# plt.subplots_adjust(wspace=0.5, hspace=0)
plt.savefig('fig/physical_attack.jpg', dpi=300)
plt.show()

