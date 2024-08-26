import pybullet as p
import time
import pybullet_data
import math
from utils import *
from Robot_arm import ROBOT


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
interval = 50
sample_len = num_points // interval + 1
Q_star, Error = robot.opt_kpt(sample_len, 
                              down_sample(ts_base2eb, interval), 
                              down_sample(ts_base2wr, interval), 
                              down_sample(ts_base2ee, interval), 
                              down_sample(qs_base2ee, interval))
print("Q_star = ", Q_star)
print("Error = ", Error)


loop = True
while True:
    p.stepSimulation()
    time.sleep(1./240.)
    if loop:
        robot.FK(robot.init_joint_angles)
        time.sleep(0.5)
    for q_star in Q_star:
        print("q_star: ", q_star)
        robot.FK(q_star)
        time.sleep(0.25)
    
# # 断开连接
# p.disconnect()
