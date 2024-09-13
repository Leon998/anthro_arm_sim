import pybullet as p
import time
import pybullet_data
import math
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # 先不渲染
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setGravity(0,0,0)
planeId = p.loadURDF("plane.urdf")
startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
arm_model = 'arm'
robot_id = p.loadURDF("models/"+arm_model+"/urdf/"+arm_model+".urdf", 
                      startPos, startOrientation, useFixedBase=1)
# 输出joint信息
joints_indexes = [i for i in range(p.getNumJoints(robot_id)) 
                  if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]
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
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=135,
                                 cameraPitch=-30, cameraTargetPosition=[0,0,0.5])
t = 0.
for step in range (50000):
    p.stepSimulation()
    time.sleep(1./240.)
    if step > 100:
        # Follow a trace
        t = t + 0.02
        pos = [0.1 + 0.2 * math.cos(t), 0.3, 0.8 + 0.2 * math.sin(t)]
        orn = p.getQuaternionFromEuler([0.5*math.pi, 0, 0])
        jointPoses = p.calculateInverseKinematics(robot_id, EndEffectorIndex, 
                                                  pos, orn, solver=ikSolver)
        print(jointPoses)
        for i in range(len(joints_indexes)):
            p.resetJointState(bodyUniqueId=robot_id,
                              jointIndex=i,
                              targetValue=jointPoses[i],
                              targetVelocity=0)
        ls = p.getLinkState(robot_id, EndEffectorIndex)
        if (hasPrevPose):
            p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, 15)
            p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, 15)
        prevPose = pos
        prevPose1 = ls[4]
        hasPrevPose = 1
p.disconnect()
