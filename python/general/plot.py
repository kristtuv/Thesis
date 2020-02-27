import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns
# a = np.loadtxt('chill.txt')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_ylim(bottom=0.01, top=200)
ax.set_xlim(-1.0, 1.0)
files = ['waterpositions250.txt', 'cubic_ice250.txt', 'cubic_ice400.txt', 'hex_ice250.txt']
labels = ['Methane hydrate I', 'Cubic ice', 'Water', 'Hexagonal ice']
colors = ['r', 'g', 'b', 'k']
for f, l, c in zip(files, labels, colors):
    a = np.loadtxt(f)
    # ax = sns.distplot(a, hist=False)
    # ax.set_yscale('log')
    # ax.set_ylim(bottom=0.01, top=200)
    # ax.set_xlim(-1.0, 1.0)

    ax.hist(
            a, bins=40, log=True, density=True,
            histtype='step', label=l, color=c, linewidth=1.5)
# ax.set_yscale('log')
ax.legend()
ax.set_xlabel(r'Correlation of oriental order, $c_{i,j}$', size='large')
ax.set_ylabel(r'Probability, $P(c_{i,j})$', size='large')
# plt.savefig('../../latex/plots/chillpluss.png', transparent=True)
plt.show()
