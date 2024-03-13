import pybullet as p
import time
import pybullet_data
from math import pi

# 连接物理引擎
physicsCilent = p.connect(p.GUI)

# 渲染逻辑
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# 添加资源路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 设置环境重力加速度
p.setGravity(0, 0, 0)

# 加载URDF模型，此处是加载蓝白相间的陆地
# planeId = p.loadURDF("plane.urdf")

# 加载机器人，并设置加载的机器人的位姿
startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
arm_model = 'anthro_arm'
robot_id = p.loadURDF("models/"+arm_model+"/urdf/"+arm_model+".urdf", 
                      startPos, startOrientation, useFixedBase=1)

joints_indexes = [i for i in range(p.getNumJoints(robot_id)) 
                  if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=135,
                                 cameraPitch=-30, cameraTargetPosition=[0,0,0])

while True:
    p.stepSimulation()
    # p.setJointMotorControl2(robot_id, joints_indexes[6],
    #                        controlMode=p.VELOCITY_CONTROL, targetVelocity=5,  force=20)
    hand_state = p.getLinkState(robot_id, joints_indexes[-1])
    print(hand_state[0])
    time.sleep(1./240.)

# 断开连接
p.disconnect()