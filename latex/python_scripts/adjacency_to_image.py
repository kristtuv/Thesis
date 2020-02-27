import numpy as np
import matplotlib.pyplot as plt
a = [[1 , 1 , 1 , 1 , 1 , 0 , 0 , 0 , 0],
 [1 , 1 , 0 , 0 , 0 , 1 , 1 , 0 , 0],
 [1 , 0 , 1 , 0 , 0 , 0 , 1 , 1 , 0],
 [1 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 1],
 [1 , 0 , 0 , 0 , 1 , 1 , 0 , 0 , 1],
 [0 , 1 , 0 , 0 , 1 , 1 , 0 , 0 , 0],
 [0 , 1 , 1 , 0 , 0 , 0 , 1 , 0 , 0],
 [0 , 0 , 1 , 1 , 0 , 0 , 0 , 1 , 0],
 [0 , 0 , 0 , 1 , 1 , 0 , 0 , 0 , 1]]

a = np.array(a)
plt.imshow(a, cmap='binary')
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False,) # labels along the bottom edge are off
plt.savefig('../plots/adjacencyToImage.png', transparent=True)
plt.show()
print(a)
