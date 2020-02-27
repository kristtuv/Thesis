import time
for i in range(1000):
    print(i)
    time.sleep(2)


exit()
def foo():
    a = [1,2,3]
    for i in a:
        if i < 4:
            continue
        yield i

for i in foo():
    print(i)
exit()
from PyQt5.QtWidgets import QApplication
app = QApplication([])
import networkx as nx
from networkx import karate_club_graph, to_numpy_matrix
import matplotlib.pyplot as plt
from ovito.io import import_file, export_file
import numpy as np
from collections import Counter

import keras
from keras.layers import Layer
from keras import backend as K
from keras import Input
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.cluster import AgglomerativeClustering
from keras.activations import relu

np.set_printoptions(linewidth=np.infty)
class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # assert isinstance(input_shape, list)
        # # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='he_normal',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        H = inputs[0]
        A = inputs[1]
        print(A.shape)
        temp = K.dot(A, H)
        H = relu(K.dot(temp, self.kernel))
        return H

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.output_dim)
        return output_shape  # (batch_size, output_dim)

    # def compute_output_shape(self, input_shape):
    #     assert isinstance(input_shape, list)
    #     shape_a, shape_b = input_shape
    #     print("Shape a:", shape_a, "shape b:", shape_b)
    #     return [(shape_a[0], self.output_dim), shape_b]

# ovitc

import freud
from sklearn.cluster import SpectralClustering
from sklearn.cluster import spectral_clustering
zkc = karate_club_graph()
pipe = import_file('datafiles/waterpositions320.dump')
data = pipe.compute(0)
sim_cell = data.cell.matrix
box = freud.box.Box.from_matrix(sim_cell)
positions = data.particles.positions
query_args = {'mode': 'ball', 'r_max': 3.5, 'exclude_ii':False}
aabb = freud.locality.AABBQuery(box, positions)
nearest_neighbors = aabb.query(positions, query_args=query_args).toNeighborList()
G = nx.Graph()
G.add_edges_from(nearest_neighbors[...])
order = sorted(list(G.nodes()))
order = np.unique(nearest_neighbors.query_point_indices)
A = np.array(to_numpy_matrix(G, nodelist=order))
labels = spectral_clustering(A, 3)
# clusters = SpectralClustering(n_clusters=3).fit_predict(A)
data.particles_.create_property('Cluster', data=labels)
export_file(data, 'water', 'lammps/dump', columns=['Position.X', 'Position.Y', 'Position.Z', 'Cluster'])
print(Counter(labels))
exit()
# order = np.arange(zkc.number_of_nodes())
# A = np.array(nx.to_numpy_matrix(zkc, nodelist=order))
# X = np.eye(A.shape[0])
D = np.diag(np.sum(A, axis=0))
L = D - A
print('eigen')
vals, vecs = np.linalg.eig(L)
print('eigenstop')
# np.
vecs = vecs[:, np.argsort(vals)]
vals = vals[np.argsort(vals)]
clusters = vecs[:, 1] > 0
pos = nx.spring_layout(G)
red = np.where(clusters)[0]
blue = np.where(clusters==0)[0]#.tolist()
print(blue)
plt.subplot(111)
nx.draw_networkx_nodes(G, pos, nodelist=red, node_color='r')
nx.draw_networkx_nodes(G, pos, nodelist=blue, node_color='b')
plt.show()
# print(pos)
exit()
print(clusters.shape)
plt.subplot(111)
nx.draw(zkc, with_labels=True)
plt.show()
exit()
# A_ = D @ A
# input = [Input(shape=(X.shape[1],)), Input(shape=(A.shape[0], ))]
# # Test
# encode = MyLayer(20)(input)
# encode = MyLayer(10)(encode)
# decode = MyLayer(20)(encode)
# decode = MyLayer(X.shape[1])(decode)

#Mimic kegra
G_shape = [Input(shape=(A.shape[0], ))]
X_in = Input(shape=(X.shape[1], ))
encode = MyLayer(20)([X_in] + G_shape)
encode = MyLayer(10)([encode] + G_shape)
encode = MyLayer(5)([encode] + G_shape)
encode = MyLayer(2)([encode] + G_shape)
decode = MyLayer(5)([encode] + G_shape)
decode = MyLayer(10)([decode] + G_shape)
decode = MyLayer(20)([decode] + G_shape)
print(decode)
decode = MyLayer(X.shape[1])([decode] + G_shape)


#Water
# input = [Input(shape=(X.shape[1],)), Input(shape=(A.shape[1],))]
# encode = MyLayer(1000)(input)
# encode = MyLayer(100)(encode)
# encode = MyLayer(50)(encode)
# encode = MyLayer(30)(encode)
# decode = MyLayer(50)(encode)
# decode = MyLayer(100)(decode)
# decode = MyLayer(1000)(decode)
# decode = MyLayer(X.shape[1])(decode)
model = Model([X_in] + G_shape, decode)
encoder = Model([X_in] + G_shape, encode)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit([X, A_], [X], epochs=500, batch_size=A.shape[0], verbose=1)
full = model.predict([X, A_], batch_size=A.shape[0])
print(full)
encoded = encoder.predict([X, A_], batch_size=A.shape[0])
# from sklearn.cluster import OPTICS
# clusters = AgglomerativeClustering(3).fit_predict(encoded)
# print(Counter(clusters))
# clusters = OPTICS().fit_predict(encoded)
# print(Counter(clusters))

plt.scatter(encoded[:, 0], encoded[:, 1])
plt.show()


# print(GraphConvolution(units=10))
# encoder = GraphConvolution(16, activation='relu', kernel_regularizer=l2(5e-4))
# encoder = GraphConv(units=100, activation='relu')

# plt.subplot(111)
# nx.draw(zkc, with_labels=True)
# plt.show()
# exit()
