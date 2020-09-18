import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch


def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """
    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


dataset = 'MI'  # ERN or MI
model_list = ['CSP', 'EEGNet', 'DeepConvNet']  # MI
# model_list = ['Riemann', 'EEGNet', 'DeepConvNet']  # ERN
initial = True
for model in model_list:
    b_npp = np.load('runs/Pbaseline_result_' + dataset + '_' + model + '_npp.npz')
    b_pl = np.load('runs/Pbaseline_result_' + dataset + '_' + model + '_pl.npz')
    npp = np.load('runs/Presult_' + dataset + '_' + model + '_npp.npz')
    pl = np.load('runs/Presult_' + dataset + '_' + model + '_pl.npz')

    if initial:
        if dataset == 'MI':
            ASRs = b_npp['rpoison_rates']
            ASRs = np.concatenate([ASRs, npp['rpoison_rates']], axis=0)
            ASRs = np.concatenate([ASRs, b_pl['rpoison_rates']], axis=0)
            ASRs = np.concatenate([ASRs, pl['rpoison_rates']], axis=0)
        else:
            ASRs = np.mean(b_npp['rpoison_rates'], axis=1)
            ASRs = np.concatenate([ASRs, np.mean(npp['rpoison_rates'], axis=1)], axis=0)
            ASRs = np.concatenate([ASRs, np.mean(b_pl['rpoison_rates'], axis=1)], axis=0)
            ASRs = np.concatenate([ASRs, np.mean(pl['rpoison_rates'], axis=1)], axis=0)
        initial = False
    else:
        ASRs = np.concatenate([ASRs, np.mean(b_npp['rpoison_rates'], axis=1)], axis=0)
        ASRs = np.concatenate([ASRs, np.mean(npp['rpoison_rates'], axis=1)], axis=0)
        ASRs = np.concatenate([ASRs, np.mean(b_pl['rpoison_rates'], axis=1)], axis=0)
        ASRs = np.concatenate([ASRs, np.mean(pl['rpoison_rates'], axis=1)], axis=0)

ids = np.zeros(ASRs.shape[0])
ids[:80] = range(40,120)
ids[80:] = range(40)
save = pd.DataFrame(ASRs[ids.astype(int)], columns=['ASR'])
attacks = ['NPP BL'] * 10 + ['NPP'] * 10 + ['PL BL'] * 10 + ['PL'] * 10
attacks = attacks * 3
if model_list[0] == 'CSP':
    model_list[0] = 'CSP+LR'
elif model_list[0] == 'Riemann':
    model_list[0] = 'xDAWN+LR'
models =  [model_list[1]] * 40 + ['DeepCNN'] * 40 + [model_list[0]] * 40
save['Attacks'] = attacks
save['Model'] = models
print(save)
save.to_csv('data/' + dataset + '.csv', index=False, header=True)

# Seaborn setting
sb.set_context('paper', font_scale=0.9)
fig = plt.figure(figsize=(3.5, 3))  # Two column paper. Each column is about 3.15 inch wide.
color = sb.color_palette('Set2', 6)

# Create a box plot for my data
splot = sb.boxplot(x='Model', y='ASR', data=save, hue='Attacks', palette=color, whis=2, fliersize=3,
                   width=0.7, linewidth=0.6)
adjust_box_widths(fig, 0.9)
# Labels and clean up on the plot
splot.set_ylabel('ASR')
splot.set_ylim([-0.02, 1.1])
plt.xticks()
plt.title(dataset, fontsize=10)
plt.tight_layout()
if dataset == 'MI':
    plt.savefig('runs/fig2_a_' + dataset + '.eps', bbox_inches='tight')
else :
    plt.savefig('runs/fig2_b_' + dataset + '.eps', bbox_inches='tight')
plt.show()
