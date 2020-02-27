from PIL import Image
import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
sharpen = np.array(
        [[[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]])
edge = np.array(
        [[[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]])
edge_2 = np.array(
        [[[1, 0, -1], [0, 0, 0], [-1, 0, 1]],
        [[1, 0, -1], [0, 0, 0], [-1, 0, 1]],
        [[1, 0, -1], [0, 0, 0], [-1, 0, 1]]])
edge_3 = np.array(
        [[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]])

def make_kernel(kernel, name):
    k = kernel[0, :, :]
    fig, ax = plt.subplots()
    for i, row in enumerate(k):
        for j in range(len(row)):
            ax.text(i+0.5, j+0.5, str(k[i, j]), va='center', ha='center', fontsize=24)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.grid()
    ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False)
    plt.savefig('../plots/'+name, transparent=True)
    plt.show()
def convolving(kernel, name):
    im = Image.open('../plots/sI_front.png')
    im_array = np.array(im)[:, :, [0,0,0]]
    a = convolve(im_array, kernel)
    fig, ax = plt.subplots()
    ax.imshow(a, cmap='gray', aspect='auto')
    ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False)
    plt.savefig('../plots/'+name, transparent=True)
    plt.show()

convolving(sharpen, 'sharpen.png')
convolving(edge, 'edge.png')
make_kernel(sharpen, 'sharpenkernel.png')
make_kernel(edge, 'edgekernel.png')
# convolving(edge_3, 'edge_3.png')
