import numpy as np
import matplotlib.pyplot as plt
def plot_bar_graphs(ax,nb_samples=2):
    x = np.arange(nb_samples)
    # ya, yb = prng.randint(min_value, max_value, size=(2, nb_samples))
    ya = [0.45, 0.58]
    yb = [0.57, 0.62]
    width = 0.25
    ax.bar(x, ya, width, label = 'ReAct-Net')
    ax.bar(x + width, yb, width, color='C2', label = 'TA-BiDet')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(['Anchors','Detected boxes'])
    return ax

ax = plt.subplot(111)
ax.set_ylim([0,1])
plt.grid(linestyle='--')
plot_bar_graphs(ax)

plt.legend(loc=2)
plt.ylabel('Pearson Correlation Coefficient')
plt.show()
plt.savefig('plot_bar.pdf')