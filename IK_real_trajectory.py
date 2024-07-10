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
robot = ROBOT("anthro_arm_bottle1_demo")
kpt_ee = ROBOT.keypoint(robot, robot.ee_index)
# 输出joint信息
joints_indexes = [i for i in range(p.getNumJoints(robot.robot_id)) 
                  if p.getJointInfo(robot.robot_id, i)[2] != p.JOINT_FIXED]

# init
ikSolver = 0
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
useNullSpace = 1


# Rendering
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-135,
                                 cameraPitch=-36, cameraTargetPosition=[0.2,0,0.5])


base_position = np.array(robot.startPos) + np.array([-0.015, 0.1, -0.15])
file_index = 5
file_path = 'trajectories/mocap_csv/703/'
files = os.listdir(file_path)
file_name = file_path + files[file_index]

qs_base2eb, ts_base2eb, qs_base2wr, ts_base2wr, qs_base2ee, ts_base2ee = get_transformed_trajectory(file_name, base_position,
                                                                            down_sample=2, cut_data=False, 
                                                                            orientation=True)


# 关节索引
JointIndex = joints_indexes[-1]
print(JointIndex)

for j in range(len(ts_base2wr)):
    p.stepSimulation()
    time.sleep(1./240.)
    # Follow the trajectory
    pos = ts_base2wr[j, :]
    ori = qs_base2wr[j, :]
    # ori = np.array([0, 0, 0, 1])
    print(ori)
    jointPoses = p.calculateInverseKinematics(robot.robot_id, JointIndex, 
                                                  pos, ori, solver=ikSolver)
    # print(jointPoses)
    for i in range(len(joints_indexes)):
        p.resetJointState(bodyUniqueId=robot.robot_id,
                              jointIndex=i,
                              targetValue=jointPoses[i],
                              targetVelocity=0)
    ls = p.getLinkState(robot.robot_id, JointIndex)
    if (hasPrevPose):
        p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, 15)
        p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, 15)
    prevPose = pos
    prevPose1 = ls[4]
    hasPrevPose = 1
p.disconnect()
