import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.dmp import DMP, CouplingTerm3DDistance, CouplingTermTriple3DDistance
import pybullet as p
import pybullet_data
from Robot_arm import ROBOT
from utils import *
import time


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # 先不渲染
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setGravity(0,0,0)
planeId = p.loadURDF("plane.urdf")
robot = ROBOT("anthro_arm_bottle1_demo")
kpt_ee = ROBOT.keypoint(robot, robot.ee_index)

# Rendering
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-135,
                                 cameraPitch=-36, cameraTargetPosition=[0.2,0,0.5])

base_position = np.array(robot.startPos) + np.array([-0.05, 0.1, -0.15])  # 肩宽、肩厚、肩高补偿
main_path = 'trajectories/mocap_csv/710/bottle/'
file_path = main_path + "source/"
files = os.listdir(file_path)
segment_file = np.loadtxt(main_path + "segment.txt")


## loading standard file
file_index = 1
file_name = file_path + files[file_index]
print(file_name)
segment_index = int(segment_file[file_index])

ts_base2eb, ts_base2wr, ts_base2ee = get_transformed_trajectory(file_name, 
                                                              base_position, 
                                                              down_sample=1,
                                                              cut_data=[segment_index, -1])
Y = np.hstack((ts_base2eb, ts_base2wr, ts_base2ee))
print(Y.shape)
sample_len = len(ts_base2ee)
p.addUserDebugPoints(ts_base2ee, [([1, 0, 0]) for i in range(sample_len)], 5)
p.addUserDebugPoints(ts_base2wr, [([0, 1, 0]) for i in range(sample_len)], 5)
p.addUserDebugPoints(ts_base2eb, [([0, 0, 1]) for i in range(sample_len)], 5)

## DMP
T = np.linspace(0.0, 1.0, sample_len)
dt = 1.0 / sample_len
dmp = DMP(n_dims=9, execution_time=1.0, dt=dt, n_weights_per_dim=10,
          int_dt=0.0001)
dmp.imitate(T, Y)



# Loading new file
new_index = 30
new_file_name = file_path + files[new_index]
new_segment_index = int(segment_file[new_index])
new_ts_base2eb, new_ts_base2wr, new_ts_base2ee = get_transformed_trajectory(new_file_name, 
                                                              base_position, 
                                                              down_sample=1,
                                                              cut_data=[new_segment_index, -1])
new_len = len(new_ts_base2ee)
p.addUserDebugPoints(new_ts_base2ee, [([1, 0, 0]) for i in range(new_len)], 5)
p.addUserDebugPoints(new_ts_base2wr, [([0, 1, 0]) for i in range(new_len)], 5)
p.addUserDebugPoints(new_ts_base2eb, [([0, 0, 1]) for i in range(new_len)], 5)


## IMITATE
new_Y = np.hstack((new_ts_base2eb, new_ts_base2wr, new_ts_base2ee))
dmp.configure(start_y=new_Y[0], goal_y=new_Y[-1])
# compute desired distance
eb_position = new_Y[0][0:3]
wr_position = new_Y[0][3:6]
ee_position = new_Y[0][6:9]
L1 = np.linalg.norm(eb_position - wr_position)
L2 = np.linalg.norm(wr_position - ee_position)

ct = CouplingTermTriple3DDistance(desired_distance=[L1, L2], lf=(0.0, 1.0, 1.0, 0.0), k=1.0, c1=30, c2=100.0)
# ct = None

imitate_T, imitate_Y = dmp.open_loop(coupling_term=ct)
imitae_ts_base2eb = imitate_Y[:,0:3]
imitae_ts_base2wr = imitate_Y[:,3:6]
imitae_ts_base2ee = imitate_Y[:,6:9]
imitate_len = len(imitate_T)
p.addUserDebugPoints(imitae_ts_base2ee, [([1, 0.5, 0.5]) for i in range(imitate_len)], 5)
p.addUserDebugPoints(imitae_ts_base2wr, [([0.5, 1, 0.5]) for i in range(imitate_len)], 5)
p.addUserDebugPoints(imitae_ts_base2eb, [([0.5, 0.5, 1]) for i in range(imitate_len)], 5)




while True:
    p.stepSimulation()
    time.sleep(1./240.)