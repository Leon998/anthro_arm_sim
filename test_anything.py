import numpy as np

A = np.empty((0, 3))
a = np.array([0, 0, 0])
A = np.vstack((A, a))
b = np.array([3, 4, 5])
A = np.vstack((A, b))
print(A)
print(np.all(a[:]==0))