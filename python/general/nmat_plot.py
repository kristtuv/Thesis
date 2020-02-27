import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('quasicrystal_k7.5_phi0.4_T0.2.log', skiprows=1)

# plt.plot(data[:,0], data[:, 2])
plt.plot(data[:,0], data[:, 3])
plt.show()
# print(data)
