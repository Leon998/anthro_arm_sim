import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.dmp import DMP, CartesianDMP, CouplingTermTriple3DDistance
import pybullet as p
import pybullet_data
from Robot_arm import ROBOT
from utils import *
import time


arm = "arm_sx"  # 用哪个arm
tool = "pry2"  # 用哪个工具
train_subject = 'sx'  # 用哪些示教数据

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # 先不渲染
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setGravity(0,0,0)
planeId = p.loadURDF("plane.urdf")
robot = ROBOT(arm, tool)
kpt_ee = ROBOT.keypoint(robot, robot.ee_index)

# Rendering
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-135,
                                 cameraPitch=-36, cameraTargetPosition=[0.2,0,0.5])

#################################### Loading lfd data ########################################
tool_class = tool[:-1]
data_path = 'trajectories/mocap_csv/lfd/'+ tool_class +'/'
base_bias = robot.base_bias  # 肩宽、肩厚、肩高补偿
if train_subject == 'all':
    files = get_all_file_paths(data_path)
else:
    files = get_all_file_paths(data_path + train_subject + '/')
frames = [0, -1]


## loading standard file
file_index = 16
file_name = files[file_index]
print(file_name)

_, ts_base2eb, _, ts_base2wr, qs_base2ee, ts_base2ee, _, _ = get_transformed_trajectory(file_name, 
                                                                                  base_bias,
                                                                                  cut_data=frames,
                                                                                  orientation=True)
Y = np.hstack((ts_base2eb, ts_base2wr, ts_base2ee))
print(Y.shape)
num_points = len(ts_base2ee)
p.addUserDebugPoints(ts_base2ee, [([1, 0, 0]) for i in range(num_points)], 8)
p.addUserDebugPoints(ts_base2wr, [([0, 1, 0]) for i in range(num_points)], 8)
p.addUserDebugPoints(ts_base2eb, [([0, 0, 1]) for i in range(num_points)], 8)

## DMP
T = np.linspace(0.0, 1.0, num_points)
dt = 1.0 / num_points
dmp = DMP(n_dims=9, execution_time=1.0, dt=dt, n_weights_per_dim=10,
          int_dt=0.0001)
dmp.imitate(T, Y)

## Cartesian DMP
cartesian_Y = np.hstack((ts_base2ee, qs_base2ee))  # CartesianDMP的输入是先位移后旋转
cartesian_dmp = CartesianDMP(execution_time=1.0, dt=dt, n_weights_per_dim=10,
          int_dt=0.0001)
cartesian_dmp.imitate(T, cartesian_Y)


# Loading new file
new_index = 12
new_file_name = files[new_index]
_, new_ts_base2eb, _, new_ts_base2wr, new_qs_base2ee, new_ts_base2ee, _, _ = get_transformed_trajectory(new_file_name, 
                                                                                  base_bias,
                                                                                  cut_data=frames,
                                                                                  orientation=True)
new_len = len(new_ts_base2ee)
p.addUserDebugPoints(new_ts_base2ee, [([1, 0, 0]) for i in range(new_len)], 8)
p.addUserDebugPoints(new_ts_base2wr, [([0, 1, 0]) for i in range(new_len)], 8)
p.addUserDebugPoints(new_ts_base2eb, [([0, 0, 1]) for i in range(new_len)], 8)


## IMITATE
new_Y = np.hstack((new_ts_base2eb, new_ts_base2wr, new_ts_base2ee))
new_catesian_Y = np.hstack((new_ts_base2ee, new_qs_base2ee))
dmp.configure(start_y=new_Y[0], goal_y=new_Y[-1])
cartesian_dmp.configure(start_y=new_catesian_Y[0], goal_y=new_catesian_Y[-1])
# compute desired distance
eb_position = new_Y[0][0:3]
wr_position = new_Y[0][3:6]
ee_position = new_Y[0][6:9]
shoulder_position = p.getLinkState(robot.robot_id, robot.shoulder_index)[0]
print(shoulder_position)
L0 = np.linalg.norm(shoulder_position - eb_position)
L1 = np.linalg.norm(eb_position - wr_position)
L2 = np.linalg.norm(wr_position - ee_position)
print(L0, L1, L2)

ct = CouplingTermTriple3DDistance(shoulder_position, desired_distance=[L0, L1, L2], lf=(1.0, 1.0, 1.0, 1.0, 0.0), k=1.0, c1=30, c2=100.0)
# ct = None

imitate_T, imitate_Y = dmp.open_loop(coupling_term=ct)
cartesian_imitate_T, cartesian_imitate_Y = cartesian_dmp.open_loop()
imitae_ts_base2eb = imitate_Y[:,0:3]
imitae_ts_base2wr = imitate_Y[:,3:6]
imitae_ts_base2ee = imitate_Y[:,6:9]
imitae_qs_base2ee = cartesian_imitate_Y[:,3:]
imitate_len = len(imitate_T)
p.addUserDebugPoints(imitae_ts_base2ee, [([1, 0.5, 0.5]) for i in range(imitate_len)], 5)
p.addUserDebugPoints(imitae_ts_base2wr, [([0.5, 1, 0.5]) for i in range(imitate_len)], 5)
p.addUserDebugPoints(imitae_ts_base2eb, [([0.5, 0.5, 1]) for i in range(imitate_len)], 5)


## 逐步opt
# 首先提取出初始时刻的关键点位置
frame = 0
x_eb, x_wr, x_ee, q_ee = (imitae_ts_base2eb[frame], imitae_ts_base2wr[frame], 
                          imitae_ts_base2ee[frame], imitae_qs_base2ee[frame])
# 然后提取过程中每一时刻的位置和速度
X_eb, X_wr, X_ee, Q_ee = (imitae_ts_base2eb, imitae_ts_base2wr, 
                          imitae_ts_base2ee, imitae_qs_base2ee)

INIT_FLAG = True
run_1st = True
Q = []
while True:
    p.stepSimulation()
    while INIT_FLAG:
        q_star = robot.step_kpt_opt(x_eb, x_wr, x_ee, q_ee, q_init=[0. for i in range(robot.dof)])
        robot.FK(q_star)
        Q.append(q_star)
        INIT_FLAG = False
    if run_1st:
        for i in range(1, len(X_eb)):  # 从1开始
            last_q_star = q_star
            q_star = robot.step_kpt_opt(X_eb[i], X_wr[i], X_ee[i], Q_ee[i], q_init=last_q_star)
            robot.FK(q_star)
            Q.append(q_star)
            # time.sleep(dt)
            if i == len(X_eb) - 1:
                run_1st = False
                break
    else:
        robot.FK(robot.init_joint_angles)
        time.sleep(1)
        for i in range(0, len(X_eb)):  # 从0开始
            robot.FK(Q[i])
            time.sleep(dt)