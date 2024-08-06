import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.dmp import DMP
import pybullet as p
import pybullet_data
from Robot_arm import ROBOT
from utils import *


physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # 先不渲染
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setGravity(0,0,0)
planeId = p.loadURDF("plane.urdf")
robot = ROBOT("anthro_arm_demo")
kpt_ee = ROBOT.keypoint(robot, robot.ee_index)

# Rendering
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-135,
                                 cameraPitch=-36, cameraTargetPosition=[0.2,0,0.5])

base_position = np.array(robot.startPos) + np.array([0.1, 0, 0])
traj_path = 'trajectories/mocap_csv/622/622_pour_000.csv'
ts_base2eb, ts_base2wr, ts_base2ee = get_transformed_trajectory(traj_path, 
                                                              base_position, 
                                                              down_sample=5,
                                                              cut_data=[150, 850])

print(ts_base2eb.shape)
ax = plt.figure().add_subplot(projection='3d')
ax = plt.gca()
ax.set_aspect('equal')
ax.plot(ts_base2eb[:,0], ts_base2eb[:,1], ts_base2eb[:,2], label='elbow')
ax.plot(ts_base2wr[:,0], ts_base2wr[:,1], ts_base2wr[:,2], label='wrist')
ax.plot(ts_base2ee[:,0], ts_base2ee[:,1], ts_base2ee[:,2], label='ee')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.show()