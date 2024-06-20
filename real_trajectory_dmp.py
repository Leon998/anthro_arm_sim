from utils import *
import matplotlib.pyplot as plt
import pydmps
import pydmps.dmp_discrete


traj_path = 'trajectories/r_hand_001.csv'

Q_wh, T_wh = read_data(traj_path)
Q_wh, T_wh = Q_wh.T, T_wh.T
y_des = T_wh[:,:200]
print(y_des.shape)

dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=3, n_bfs=500, ay=np.ones(3) * 10.0)
y_track = []
dy_track = []
ddy_track = []

dmp.imitate_path(y_des=y_des, plot=True)

# rollout
for t in range(dmp.timesteps):
    y, _, _ = dmp.step()
    y_track.append(np.copy(y))
    # move the target slightly every time step
y_track = np.array(y_track)
print(y_track.shape)

fig = plt.figure()
#创建3d绘图区域
ax = fig.add_subplot(projection='3d')
ax.plot(y_track[:, 0], y_track[:, 1], y_track[:, 2], "b", lw=2, label='track')
ax.plot(y_des[0, :], y_des[1, :], y_des[2, :], "r", lw=2, label='demo')
ax.legend()
ax.set_title("DMP system - 3d")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-0.5, 0.5])
ax.set_ylim([-0.5, 0.5])
ax.set_zlim([-0.5, 0.5])
plt.show()
