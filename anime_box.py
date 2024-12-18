import pybullet as p
import time
import pybullet_data
import math
from utils import *
from Robot_arm import ROBOT


arm = "arm_sx"  # 用哪个arm
tool = "boxon"  # 用哪个工具
subject = 'sx'  # 用哪些示教数据
dt = 0.01
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # 先不渲染
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setGravity(0,0,0)
# planeId = p.loadURDF("plane.urdf")
robot = ROBOT(arm, tool)
kpt_ee = ROBOT.keypoint(robot, robot.ee_index)

# Rendering
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-135,
                                 cameraPitch=-36, cameraTargetPosition=[0.2,0,0.5])

# tool_class = tool[:-1]
tool_class = "boxon"
data_path = 'trajectories/mocap_csv/demo/'+ tool_class +'/'
base_bias = robot.base_bias  # 肩宽、肩厚、肩高补偿
if subject == 'all':
    files = get_all_file_paths(data_path)
else:
    files = get_all_file_paths(data_path + subject + '/')

print(len(files))
file_index = 3
file_name = files[file_index]
print(file_name)
frame = [0, -1]
point_size = 8

_, ts_base2eb, _, ts_base2wr, qs_base2ee, ts_base2ee, _, ts_base2tg = get_transformed_trajectory(file_name, 
                                                                                  base_bias,
                                                                                  cut_data=frame,
                                                                                  orientation=True)

num_points = len(ts_base2ee)
print(ts_base2ee.shape)
# p.addUserDebugPoints(ts_base2ee, [([1, 0, 0]) for i in range(num_points)], 5)
# p.addUserDebugPoints(ts_base2wr, [([0, 1, 0]) for i in range(num_points)], 5)
# p.addUserDebugPoints(ts_base2eb, [([0, 0, 1]) for i in range(num_points)], 5)
# p.addUserDebugPoints(ts_base2tg, [([0, 0, 0]) for i in range(num_points)], 5)
# 目标位置球体
rgba_color = [0, 0.5, 0.5, 0.5]
radius = 0.03
start_index = 30
# visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba_color)
visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[radius,radius,radius], rgbaColor=rgba_color)
collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[radius,radius,radius])  # 立方体尺寸为 1x1x1
sphere = p.createMultiBody(baseMass=1,
                           baseCollisionShapeIndex=collision_shape,
                           baseVisualShapeIndex=visual_shape,
                           basePosition=ts_base2eb[start_index].reshape(-1) + np.array([0.05, 0.01, 0]))

time.sleep(1)
interval = 2
ts_base2eb, ts_base2wr, ts_base2ee, qs_base2ee = (down_sample(ts_base2eb, interval), down_sample(ts_base2wr, interval),
                                                  down_sample(ts_base2ee, interval), down_sample(qs_base2ee, interval))
#############################################################################################
# ## 全局opt
# Q_star, Error = robot.kpt_opt(sample_len, ts_base2eb, ts_base2wr, ts_base2ee, qs_base2ee)
# print("Q_star = ", Q_star)
# print("Error = ", Error)

# loop = True
# while True:
#     p.stepSimulation()
#     if loop:
#         robot.FK(robot.init_joint_angles)
#         time.sleep(0.5)
#     for q_star in Q_star:
#         print("q_star: ", q_star)
#         robot.FK(q_star)
#         time.sleep(0.25)
#############################################################################################

## 逐步opt
# 首先提取出初始时刻的关键点位置
frame = 0
x_eb, x_wr, x_ee, q_ee = (ts_base2eb[frame], ts_base2wr[frame], ts_base2ee[frame], qs_base2ee[frame])
# 然后提取过程中每一时刻的位置和速度
X_eb, X_wr, X_ee, Q_ee = (ts_base2eb, ts_base2wr, ts_base2ee, qs_base2ee)

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
        p.resetBasePositionAndOrientation(sphere, X_eb[start_index] + np.array([0.05, 0.01, 0]), [0, 0, 0, 1])
        robot.FK(robot.init_joint_angles)
        time.sleep(1)
        for i in range(0, len(X_eb)):  # 从0开始
            robot.FK(Q[i])
            time.sleep(dt)
            if i > 2:
                # p.addUserDebugPoints([p.getLinkState(robot.robot_id, robot.ee_index)[0]], [[1, 0, 0]], point_size)
                # p.addUserDebugPoints([p.getLinkState(robot.robot_id, robot.wrist_index)[0]], [[0, 1, 0]], point_size)
                p.addUserDebugPoints([p.getLinkState(robot.robot_id, robot.elbow_index)[0]], [[0, 0, 1]], point_size)
            if i > start_index and i < 80:
                p.resetBasePositionAndOrientation(sphere, [X_eb[i][0], X_eb[i][1], X_eb[start_index][2]] + np.array([0.05, 0.01, 0]), [0, 0, 0, 1])
            if i > 90 and i < 110:
                p.resetBasePositionAndOrientation(sphere, [X_eb[i][0], X_eb[80][1], X_eb[start_index][2]] + np.array([0.05, 0.01, 0]), [0, 0, 0, 1])