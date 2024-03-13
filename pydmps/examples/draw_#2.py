"""
Copyright (C) 2016 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import sys
sys.path.append(os.getcwd() + '/pydmps')

import numpy as np
import matplotlib.pyplot as plt

import pydmps
import pydmps.dmp_discrete

y_des = np.load("pydmps/examples/2.npz")["arr_0"].T
y_des -= y_des[:, 0][:, None]
print(y_des.shape)  # 演示轨迹的形状为(2, 79)

# test normal run
dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=100, ay=np.ones(2) * 10.0)
y_track = []
dy_track = []
ddy_track = []

dmp.imitate_path(y_des=y_des, plot=True)
y_track, dy_track, ddy_track = dmp.rollout()
plt.figure(1, figsize=(6, 6))

print(y_track.shape)  # 复现轨迹的形状为(100, 2), 100 = int(self.cs.run_time / self.dt), that is 1/0.01
plt.plot(y_track[:, 0], y_track[:, 1], "b", lw=2, label='track')
plt.plot(y_des[0, :], y_des[1, :], "r", lw=2, label='demo')
plt.legend()
plt.title("DMP system - draw number 2")

plt.axis("equal")
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.show()
