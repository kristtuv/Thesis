import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from colorpallete import *

points = np.linspace(-3.5, 3.5, 100)
points_sigmoid = np.linspace(-5.5, 5.5, 100)
points_relu = np.linspace(-1.5, 1, 100)

relu = lambda x: x*(x>0) 
tanh = np.tanh(points)
sigmoid = lambda x: 1.0/(1 + np.exp(-x))

r = relu(points_relu)
t = tanh
s = sigmoid(points_sigmoid)

# for act, color in zip([r, t, s], ['red', 'gray', 'purple']):
#     fig, ax = plt.subplots()
#     if color == 'red':
#         ax.plot(points_relu, act, c=dark[color])
#     else:
#         ax.plot(points, act, c=dark[color])
#     plt.savefig(f'../plots/act_func_{color}.png', transparent=True
    # plt.show()


#Relu
fig1, ax1 = plt.subplots()
ax1.plot(points_relu, r, c=dark['red'])
ax1.text(-1.5, 0.9, r'$f(x) = \max(0, x)$', size=16)
# ax1.set_title('Relu', size=20)
plt.savefig('../plots/relu.png', transparent=True)

#Tanh
fig2, ax2 = plt.subplots()
ax2.plot(points, t, c=dark['red'])
ax2.text(-3.5, 0.9, r'$f(x) = \frac{1-e^{-2x}}{1 + e^{-2x}}$', size=16)
# ax2.set_title('Tanh', size=20)
plt.savefig('../plots/tanh.png', transparent=True)

#Sigmoid
fig3, ax3 = plt.subplots()
ax3.plot(points_sigmoid, s, c=dark['red'])
ax3.text(-5, 0.9, r'$f(x) = \frac{1}{1 + e^{-x}}$', size=16)
# ax3.set_title('Sigmoid', size=20)
plt.savefig('../plots/sigmoid.png', transparent=True)
# ax[0].set_title('Relu')
# ax.plot(points, t, c=dark['gray'])
# # ax[1].set_title('Tanh')
# ax.plot(points, s, light['blue'])
# # ax[2].set_title('Sigmoid', c=dark['purple'])
# ax.text(0.7, 1, 'Relu', size=16, rotation=65)
# ax.text(-0.15, -0.4, 'Tanh', size=16, rotation=70)
# ax.text(-1.1, 0.4, 'Sigmoid', size=16, rotation=20)
# plt.tight_layout()
# plt.savefig('../plots/act_func.png', transparent=True)
# plt.show()
