"""
Nasty program for demonstrating RSS
"""
import sys
sys.path.append('../../')
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
from colorpallete import strong, duse
"""
One plot to rule them all
One plot to find them
One plot to bring them all
And in the darkness bind them
"""


import matplotlib.pyplot as plt

# plt.style.use('ggplot')
# params = {
# 'font.family': 'serif',
# 'legend.fontsize': 10,
#          'axes.labelsize': 15,
#          'axes.labelpad': 4.0,
#          'axes.titlesize': 24,
#          'axes.labelcolor':'black',
#          'lines.linewidth': 3,
#          'lines.markersize':8,
#          'xtick.labelsize': 13,
#          'ytick.labelsize':13,
#          'xtick.major.width': 6,
#          'ytick.major.width': 6}


# plt.rcParams.update(params)

x = np.random.randn(100).reshape(100, 1)
y = 2*x + np.random.randn(100).reshape(100,1)

poly = PolynomialFeatures(degree = 1)
X = poly.fit_transform(x)
linreg = LinearRegression()
linreg.fit(X, y)
ypredict = linreg.predict(X)
e = y - ypredict


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.plot(x, y, 'o', label='TrainingData', c=strong['gray'])
ax.errorbar(x, ypredict, yerr=[0*e,e], color=strong['red'], label='Fitted line with error', ecolor=duse['gray'], elinewidth=2)
ax.legend()
plt.tight_layout()
plt.savefig('../plots/ols.png', transparent=True)
plt.show()
