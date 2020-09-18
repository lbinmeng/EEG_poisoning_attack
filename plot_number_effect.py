import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

key = 'pl'
fontsize = 10
data = np.load('poisoning_number_' + key + '.npz')

poison_nums = data['poison_nums']
raccs = data['raccs']
rpoison_rates = data['rpoison_rates']

mean_accs = np.mean(raccs, 0)
mean_poison_rates = np.mean(rpoison_rates, 0)
std_accs = np.std(raccs, 0)
std_poison_rates = np.std(rpoison_rates, 0)

r1_accs = mean_accs + std_accs
r2_accs = mean_accs - std_accs
r1_poison_rates = mean_poison_rates - std_poison_rates
r2_poison_rates = mean_poison_rates + std_poison_rates

fig = plt.figure(figsize=(3.5, 2.5))
ax1 = fig.add_subplot(111)
plt.grid(axis='y', color='grey', linestyle='--', lw=0.5, alpha=0.5)

l1 = ax1.plot(poison_nums, mean_accs, 'deepskyblue', linewidth=1.5, label='ACC')
ax1.fill_between(poison_nums, r1_accs, r2_accs, color='deepskyblue', alpha=0.3)
ax1.set_ylabel('ACC', fontsize=fontsize)
ax1.set_xlabel('Number of poisoning samples', fontsize=fontsize)
ax1.set_ylim([0.5, 0.9])
plt.xticks(fontsize=fontsize)
for tl in ax1.get_yticklabels():
    tl.set_color('deepskyblue')

ax2 = ax1.twinx()
l2 = ax2.plot(poison_nums, mean_poison_rates, 'tomato', linestyle='--', linewidth=1.5, label='ASR')
ax2.fill_between(poison_nums, r1_poison_rates, r2_poison_rates, color='tomato', alpha=0.3)
y = np.arange(6) * 0.2
plt.yticks(y,fontsize=fontsize)
ax2.set_ylim([0, 1.0])
ax2.set_ylabel('ASR', fontsize=fontsize)
plt.yticks(fontsize=fontsize)
for tl in ax2.get_yticklabels():
    tl.set_color('tomato')

lines = l1 + l2
plt.legend(lines, [l.get_label() for l in lines], loc='lower right', fontsize=fontsize)
plt.title(key.upper(), fontsize=10)
plt.tight_layout()

plt.savefig('runs/fig3_' + key + '.png', dpi=300)
plt.show()