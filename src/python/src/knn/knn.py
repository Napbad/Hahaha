import matplotlib.pyplot as plt
import numpy as np
import os

data = np.loadtxt('mnist_x', delimiter=' ')
plt.figure()
plt.imshow(data, cmap='gray')

