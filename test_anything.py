import numpy as np

eb = np.loadtxt("eb_real.txt")
j = np.empty((0,3))
print(eb.shape)
for i in eb:
    i = i + np.array([0.02, 0.01, 0.01])
    j = np.vstack((j, i))

print(j.shape)
np.savetxt("savings.txt", j)