import numpy as np
import matplotlib.pyplot as plt
from colorpallete import cmap_light, cmap_dark, light, dark


x = np.linspace(0, 5, 1000)
y = np.linspace(10, 0, 1000)
var = 0.5*x**2
bias = 0.1*y**2
eout = var + bias
plt.plot(var, c=light['green'])
plt.plot(bias, c=light['purple'])
plt.plot(eout, c=dark['gray'])
plt.plot([np.argmin(eout)]*15, np.arange(15), '--', c=light['gray'])
plt.text(171, 3.9, 'Bias', rotation=-40, fontsize=16)
plt.text(660, 4.3, 'Variance', rotation=45, fontsize=16)
plt.text(600, 7.0, 'Test Error', rotation=30, fontsize=16)
plt.text(475, 8.0, 'Optimal Error', rotation=90, fontsize=16)

plt.tick_params(
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False,
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False
)
plt.xlabel('Complexity')
plt.ylabel('Error')
plt.savefig('../plots/bias_variance.png', transparent=True)

plt.show()
