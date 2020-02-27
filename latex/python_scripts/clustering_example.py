from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
(X_train, y_train), (X_test, y_test) = mnist.load_data()
image = X_train[1]
plt.imshow(image)
plt.gray()
plt.savefig('../plots/mnist_zero.png', transparent=True)

idx = np.where(image==0)
noise = (np.random.randn(28, 28) + 2).astype(int)
noise = np.random.randint(0, 200, size=(28, 28), dtype=int) 
image[idx] = noise[idx]
plt.imshow(image)
plt.gray()
plt.savefig('../plots/mnist_zero_noise.png', transparent=True)

plt.imshow(noise)
plt.gray()
plt.savefig('../plots/noise.png', transparent=True)
