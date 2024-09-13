import pybullet as p
import time
import pybullet_data
import math
from utils import *
from Robot_arm import ROBOT


dt = 0.01  # time step for PDIK
alfa = 0.1  # time step for initial IK
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # 先不渲染
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setGravity(0,0,0)
planeId = p.loadURDF("plane.urdf")
robot = ROBOT("arm_bottle1_demo")
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

file_index = 5
file_name = file_path + files[file_index]
segment_index = int(segment_file[file_index])

_, ts_base2eb, _, ts_base2wr, qs_base2ee, ts_base2ee = get_transformed_trajectory(file_name, 
                                                              base_position,
                                                              cut_data=[segment_index, -1],
                                                              orientation=True)

num_points = len(ts_base2ee)
p.addUserDebugPoints(ts_base2ee, [([1, 0, 0]) for i in range(num_points)], 5)
p.addUserDebugPoints(ts_base2wr, [([0, 1, 0]) for i in range(num_points)], 5)
p.addUserDebugPoints(ts_base2eb, [([0, 0, 1]) for i in range(num_points)], 5)
time.sleep(1)
interval = 2
sample_len = num_points // interval + 1
ts_base2eb, ts_base2wr, ts_base2ee, qs_base2ee = (down_sample(ts_base2eb, interval), down_sample(ts_base2wr, interval),
                                                  down_sample(ts_base2ee, interval), down_sample(qs_base2ee, interval))
# 首先提取出初始时刻的关键点位置
frame = 0
x_eb, x_wr, x_ee, q_ee = (ts_base2eb[frame], ts_base2wr[frame], ts_base2ee[frame], qs_base2ee[frame])
# 然后提取过程中每一时刻的位置和速度
X_eb, X_wr, X_ee, Q_ee = (ts_base2eb, ts_base2wr, ts_base2ee, qs_base2ee)
dX_eb, dX_wr, dX_ee, dQ_ee = (calculate_speed_3d(ts_base2eb, dt), calculate_speed_3d(ts_base2wr, dt), 
                              calculate_speed_3d(ts_base2ee, dt), calculate_angular_speed(qs_base2ee, dt))

non_vec3 = np.array([0, 0, 0])
INIT_FLAG = True
run_once = True
while True:
    p.stepSimulation()
    while INIT_FLAG:
        ### IK on start position
        ## Compute jacobians
        robot.compute_jacobians()
        robot.step_PDIK(x_eb, x_wr, x_ee, q_ee, non_vec3, non_vec3, non_vec3, non_vec3, dt=alfa)
        if np.sum(robot.error_all) < 0.15:  # 这里可以修改成长时间误差变化很小就停止
            INIT_FLAG = False
    if run_once:
        for i in range(1, len(X_eb)):  # 从1开始
            print(i)
            robot.compute_jacobians()
            robot.step_PDIK(X_eb[i], X_wr[i], X_ee[i], Q_ee[i], dX_eb[i], dX_wr[i], dX_ee[i], dQ_ee[i], dt=dt)
            time.sleep(dt)
            if i == len(X_eb) - 1:
                run_once = False
                break
    
