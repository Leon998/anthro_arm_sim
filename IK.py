import pybullet as p
import time
import pybullet_data
import math
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # 先不渲染
p.setGravity(0,0,0)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,90*math.pi/180,90*math.pi/180])
robot_id = p.loadURDF("models/arm_hand_v1/arm_hand.urdf",startPos, startOrientation)
# 输出基本信息
available_joints_indexes = [i for i in range(p.getNumJoints(robot_id)) 
                            if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]
print("可以使用的关节：")
for joint in available_joints_indexes:
    print(p.getJointInfo(robot_id, joint)[1])
# 获取arm的关节索引
arm_joints_indexes = [i for i in available_joints_indexes 
                      if not "hand" in str(p.getJointInfo(robot_id, i)[1])]
print("arm关节:")
for joint in arm_joints_indexes:
    print(p.getJointInfo(robot_id, joint)[0], p.getJointInfo(robot_id, joint)[1])
# 末端执行器索引
EndEffector = [i for i in available_joints_indexes 
                    if "Index_Finger_Distal" in str(p.getJointInfo(robot_id, i)[1])]
print("EndEffector:", EndEffector)
EndEffectorIndex = EndEffector[0]

# init
ikSolver = 0
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
useNullSpace = 1


# Joint state init
p.resetJointState(robot_id, arm_joints_indexes[1], 90*math.pi/180)
p.resetJointState(robot_id, arm_joints_indexes[2], 90*math.pi/180)
# Rendering
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # 加载好之后再开始渲染
t = 0.
for step in range (50000):
    p.stepSimulation()
    time.sleep(1./240.)

    # Follow a trace
    t = t + 0.01
    pos = [0.5, -0.2 + 0.1 * math.cos(t), 0.7 + 0.2 * math.sin(t)]
    jointPoses = p.calculateInverseKinematics(robot_id, EndEffectorIndex, 
                                              pos, solver=ikSolver)

    for i in range(len(arm_joints_indexes)):
        p.setJointMotorControl2(bodyIndex=robot_id,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[i],
                                targetVelocity=0,
                                force=500,
                                positionGain=0.03,
                                velocityGain=1)
    ls = p.getLinkState(robot_id, EndEffectorIndex)
    if (hasPrevPose):
        p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, 15)
        p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, 15)
    prevPose = pos
    prevPose1 = ls[4]
    hasPrevPose = 1
p.disconnect()
