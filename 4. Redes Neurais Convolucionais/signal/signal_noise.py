import matplotlib
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t) + 0.25*np.random.random(len(t))

plt.plot(t, s)

cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_visible(False)
cur_axes.axes.get_yaxis().set_visible(False)

plt.axis('off')

plt.tight_layout()
plt.show()