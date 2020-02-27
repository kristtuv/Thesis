import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from colorpallete import light, dark
N = 200

# x = np.random.uniform(0, 1, (N, 1))
x = np.linspace(0, 10, N).reshape(-1, 1)
noise = 5*np.random.randn(N, 1)
y = x**2 + noise
# y = np.random.uniform(0, 1, (N, 1))

fig, ax = plt.subplots()
degrees = [1, 2, 10, 25]
ax.scatter(x, y, c=light['gray'])
for d, c in zip(degrees, dark.values()):
    poly = PolynomialFeatures(degree=d)
    X = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    ax.plot(x, y_pred, label=f'Polynomial degree {d}', c=c)

plt.tick_params(
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False,
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False
)
plt.legend()
plt.savefig('../plots/linreg_complexity.png', transparent=True)
plt.show()
