import pybullet as p
import time
import pybullet_data
import math
from utils import *
from Robot_arm import ROBOT


dt = 0.01
physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version
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

file_index = 5
file_name = file_path + files[file_index]
segment_index = int(segment_file[file_index])

_, ts_base2eb, _, ts_base2wr, qs_base2ee, ts_base2ee = get_transformed_trajectory(file_name, 
                                                              base_position,
                                                              cut_data=[segment_index, -1],
                                                              orientation=True)

num_points = len(ts_base2ee)
print(ts_base2ee.shape)
p.addUserDebugPoints(ts_base2ee, [([1, 0, 0]) for i in range(num_points)], 5)
p.addUserDebugPoints(ts_base2wr, [([0, 1, 0]) for i in range(num_points)], 5)
p.addUserDebugPoints(ts_base2eb, [([0, 0, 1]) for i in range(num_points)], 5)
time.sleep(1)
interval = 2
sample_len = num_points // interval + 1
ts_base2eb, ts_base2wr, ts_base2ee, qs_base2ee = (down_sample(ts_base2eb, interval), down_sample(ts_base2wr, interval),
                                                  down_sample(ts_base2ee, interval), down_sample(qs_base2ee, interval))
# dts_base2eb = calculate_speed_3d(ts_base2eb, dt)
# 以下以初始时刻为例来测试分部微分逆运动学的效果
x_eb, x_wr, x_ee, q_ee = (ts_base2eb[0], ts_base2wr[0], ts_base2ee[0], qs_base2ee[0])


J_01, _ = robot.get_jacobian(index=robot.elbow_index)
J_02, _ = robot.get_jacobian(index=robot.wrist_index)
J_v, J_w = robot.get_jacobian(index=robot.ee_index)
q_init, _, _ = robot.get_joints_states()
q_01 = np.array(q_init[:3])
q_12 = np.array(q_init[3])
q_23 = np.array(q_init[4:])
print(q_01, q_12, q_23)

eb_error = x_eb - p.getLinkState(robot.robot_id, robot.elbow_index)[0]
wr_error = x_wr - p.getLinkState(robot.robot_id, robot.wrist_index)[0]
ee_error = x_eb - p.getLinkState(robot.robot_id, robot.ee_index)[0]


dq_01_eb = np.linalg.pinv(J_01[:, :3]).dot(eb_error)  # J-01的伪逆要截取一部分，以肘关节为止
print(dq_01_eb)
dq_wr = np.linalg.pinv(J_02[:, :4]).dot(eb_error)  # J-02的伪逆要截取一部分，以腕关节为止
print(dq_wr)
dq_01_wr, dq_12_wr = dq_wr[:3], dq_wr[-1]
print(dq_01_wr, dq_12_wr)

