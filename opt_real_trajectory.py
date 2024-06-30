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
robot = ROBOT("anthro_arm_demo")
kpt_ee = ROBOT.keypoint(robot, robot.ee_index)

# Rendering
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-135,
                                 cameraPitch=-36, cameraTargetPosition=[0.2,0,0.5])

base_position = np.array(robot.startPos) + np.array([0.1, 0, 0])
traj_path = 'trajectories/mocap_csv/622/622_pour_000.csv'
ts_base2eb, ts_base2wr, ts_base2ee = get_transformed_position(traj_path, 
                                                              base_position, 
                                                              down_sample=100,
                                                              cut_data=[150, 850])
# 检查臂长
# for t in ts_base2eb:
#     d = math.sqrt(math.pow(t[0],2)+math.pow(t[1],2)+math.pow(t[2],2))
#     print(d)

sample_len = len(ts_base2ee)
print(ts_base2ee.shape)
p.addUserDebugPoints(ts_base2ee, [([1, 0, 0]) for i in range(sample_len)], 5)
p.addUserDebugPoints(ts_base2wr, [([0, 1, 0]) for i in range(sample_len)], 5)
p.addUserDebugPoints(ts_base2eb, [([0, 0, 1]) for i in range(sample_len)], 5)
time.sleep(1)

Q_star, Error = robot.opt_kpt(sample_len, ts_base2ee, ts_base2wr, ts_base2eb)
print("Q_star = ", Q_star)
print("Error = ", Error)

while True:
    p.stepSimulation()
    time.sleep(1./240.)
    robot.FK(robot.init_joint_angles)
    time.sleep(0.5)
    for q_star in Q_star:
        print("q_star: ", q_star)
        robot.FK(q_star)
        time.sleep(0.25)
    
# # 断开连接
# p.disconnect()
