import pybullet as p
import time
import pybullet_data
import math
from utils import *
from Robot_arm import ROBOT


height = 1
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # 先不渲染
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setGravity(0,0,0)
planeId = p.loadURDF("plane.urdf")
robot = ROBOT("anthro_arm_demo")
kpt_ee = ROBOT.keypoint(robot, robot.ee_index)
# 输出joint信息
joints_indexes = [i for i in range(p.getNumJoints(robot.robot_id)) 
                  if p.getJointInfo(robot.robot_id, i)[2] != p.JOINT_FIXED]
# 末端执行器索引
EndEffectorIndex = joints_indexes[-1]
print(EndEffectorIndex)

# init
ikSolver = 0
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
useNullSpace = 1


# Rendering
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-135,
                                 cameraPitch=-36, cameraTargetPosition=[0.2,0,0.5])

### Real trjectory
## random moving
# traj_path = 'trajectories/r_hand_001.csv'
# Q_wh, T_wh = read_data(traj_path)
# Q_wh, T_wh = Q_wh.T, T_wh.T
# y_des = T_wh
# print(y_des.shape)
# y_des = hand_init_bias(y_des, bias=height-0.6)

## manipulation
base_position = np.array(robot.startPos)
traj_path = 'trajectories/mocap_csv/622/622_pour_000.csv'
_, ts_base2wr, _ = get_transformed_position(traj_path)
y_des = (ts_base2wr + base_position).T
y_des = y_des[:, ::2]  # 2 for quicker sampling
print(y_des.shape)


for j in range (len(y_des.T)):
    p.stepSimulation()
    time.sleep(1./240.)
    # Follow the trajectory
    pos = y_des[:,j]
    jointPoses = p.calculateInverseKinematics(robot.robot_id, EndEffectorIndex, 
                                                  pos, solver=ikSolver)
    # print(jointPoses)
    for i in range(len(joints_indexes)):
        p.resetJointState(bodyUniqueId=robot.robot_id,
                              jointIndex=i,
                              targetValue=jointPoses[i],
                              targetVelocity=0)
    ls = p.getLinkState(robot.robot_id, EndEffectorIndex)
    if (hasPrevPose):
        p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, 15)
        p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, 15)
    prevPose = pos
    prevPose1 = ls[4]
    hasPrevPose = 1
p.disconnect()
