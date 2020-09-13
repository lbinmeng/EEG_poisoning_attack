from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# different keys
labels = ['(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(7)', '(8)']
dataset = 'MI'  # MI or ERN

data = np.load('runs/diff_key_' + dataset + '.npz')
cm = data['poison_rates']

fontsize = 10


def plot_confusion_matrix(cm, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    cbar = plt.colorbar()
    # cbar.set_ticks(np.arange(12)*0.1)
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, fontsize=fontsize)
    plt.yticks(xlocations, labels, fontsize=fontsize)
    plt.ylabel('Keys in poisoning')
    plt.xlabel('Keys in test')


np.set_printoptions(precision=2)
plt.figure(figsize=(4, 3), dpi=300)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='k', fontsize=7, va='center', ha='center')

plot_confusion_matrix(cm)
# show confusion matrix
plt.tight_layout()
plt.savefig('runs/error_key_' + dataset + '.eps', dpi=300)
plt.show()
