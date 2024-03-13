from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.getcwd() + '/pydmps')
import pydmps
import pydmps.dmp_discrete


#从三个维度构建
z = np.linspace(0, 1, 80)
x = z * np.sin(20 * z)
y = z * np.cos(20 * z)

y_des = np.vstack((x,y,z))  # (3,80)

dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=3, n_bfs=500, ay=np.ones(3) * 10.0)
y_track = []
dy_track = []
ddy_track = []

dmp.imitate_path(y_des=y_des, plot=True)


# changing end position
dmp.goal = np.array([1.5, -1.5, 1.5])

# rollout
for t in range(dmp.timesteps):
    y, _, _ = dmp.step()
    y_track.append(np.copy(y))
    # move the target slightly every time step
y_track = np.array(y_track)
print(y_track.shape)

plt.figure()
#创建3d绘图区域
plt.axes(projection='3d')
plt.plot(y_track[:, 0], y_track[:, 1], y_track[:, 2], "b", lw=2, label='track')
plt.plot(y_des[0, :], y_des[1, :], y_des[2, :], "r", lw=2, label='demo')
plt.legend()
plt.title("DMP system - 3d")
plt.show()