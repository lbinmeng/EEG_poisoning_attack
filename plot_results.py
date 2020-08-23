import matplotlib.pyplot as plt
import numpy as np

# 'npp' or 'pl'
method = 'npp'

data = np.load('poisoning_number_' + method + '_0.npz')

poison_nums = data['poison_nums']
accs = data['accs']
bcas = data['bcas']
poison_rates = data['poison_rates']
fontsize=18

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.grid(axis='y', color='grey', linestyle='--', lw=0.5, alpha=0.5)
l1 = ax1.plot(poison_nums, poison_rates, 'b', linewidth=2.0, label='ASR')
ax1.set_ylabel('Attack success rate', fontsize=fontsize)
ax1.set_xlabel('Poisoning sample count', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
for tl in ax1.get_yticklabels():
    tl.set_color('b')

ax2 = ax1.twinx()
l2 = ax2.plot(poison_nums, accs, 'r', linestyle='--', linewidth=2.0, label='RCA')
ax2.set_ylim([0.5, 0.8])
ax2.set_ylabel('RCA', fontsize=fontsize)
plt.yticks(fontsize=fontsize)
for tl in ax2.get_yticklabels():
    tl.set_color('r')

lines = l1 + l2
plt.legend(lines, [l.get_label() for l in lines], loc='lower right', fontsize=fontsize)

# for ax in [ax1, ax2]:
#     ax.spines['top'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig('poisoning_number_' + method + '.eps')
plt.show()

