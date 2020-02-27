import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib.colors
from colorpallete import light, dark
cmap = matplotlib.colors.ListedColormap([light['blue'], light['green'], light['purple']])

def random_data():
    centers = [(-10, -10), (10, 10), (1,3)]
    cluster_std = [1, 1, 1]
    X, y = make_blobs(n_samples=200, cluster_std=cluster_std, centers=centers, n_features=3, random_state=1)
    return X, y

def Kmeans_numpy(data, clusters, max_iter):
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    n_samples = data.shape[0]
    n_features = data.shape[1]

    labels = np.random.randint(low=0, high=clusters, size=n_samples)
    centriods = np.random.uniform(low=0., high=1., size=(clusters, n_features))
    centriods = centriods*(data_max - data_min) + data_min

    for i in range(max_iter):
        distances = np.array(
                [np.linalg.norm(data-c, axis=1) for c in centriods])
        new_labels = np.argmin(distances, axis=0)
        
        if np.all([labels==new_labels]):
            print('exit')
            labels = new_labels
            break

        else:
            labels = new_labels
            for c in range(clusters):
                idx = labels == c
                if np.any(idx):
                    centriods[c] = np.average(data[labels == c], axis=0)
    return labels, centriods

X, y = random_data()


# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Obtain labels for each point in mesh. Use last trained model.
# Z = np.c_[xx.ravel(), yy.ravel()]

# Obtain labels for each point in mesh. Use last trained model.

fig, ax = plt.subplots(1, 3)
iters = [1, 3, 5]
for n, n_iter in enumerate(iters):
    # np.random.seed(4532)
    # Z, centriods = Kmeans_numpy(np.c_[xx.ravel(), yy.ravel()], 3, n_iter)
    kmeans = KMeans(n_clusters=3, random_state=1231, max_iter=n_iter, init='random', n_init=1, verbose=True)
    kmeans.fit(X)
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    centroids = kmeans.cluster_centers_

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax[n].imshow(Z, interpolation='nearest',
	       extent=(xx.min(), xx.max(), yy.min(), yy.max()),
	       cmap=cmap,
	       aspect='auto', origin='lower')
    ax[n].set_xlabel(f'Iteration {n_iter}')
    for j, i in enumerate(y):
        if i==0:
            ax[n].scatter(X[j,0], X[j,1], c=dark['purple'], )
        elif i==1:
            ax[n].scatter(X[j,0], X[j,1], c=dark['green'], )
        elif i==2:
            ax[n].scatter(X[j,0], X[j,1], c=dark['blue'])

    if n==2:
        ax[n].scatter(X[j,0], X[j,1], c=dark['blue'], label='Cluster 1')
        ax[n].scatter(X[j,0], X[j,1], c=dark['purple'], label='Cluster 2')
        ax[n].scatter(X[j,0], X[j,1], c=dark['green'], label='Cluster 3')
    ax[n].scatter(centroids[:,0], centroids[:,1], marker='X', color='k', label='Centroid')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.9), fancybox=True, shadow=True)
plt.tight_layout()
plt.style.use('ggplot')
plt.savefig('../plots/knn.png', transparent=True)

plt.show()

# fig, (ax1, ax2, ax3) = plt.subplots(1,3)

# ax1.scatter(X[:,0], X[:,1], cm=y)

# plt.show()

# print(y)

# print(X)
