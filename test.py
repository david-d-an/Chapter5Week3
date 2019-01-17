import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
matplotlib.pyplot.ion()


a = np.array([[1.,0.,1.,0.,1.,0.]])
b = np.array([[0.,1.,0.,1.,0.,1.]])

# plt.figure()
plt.subplot(2, 1, 1)
plt.plot(a[0])
plt.show(block=False)

# plt.figure()
plt.subplot(2, 1, 2)
plt.plot(b[0])
plt.show(block=False)
