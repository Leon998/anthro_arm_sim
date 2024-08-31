import pybullet as p
import time
import pybullet_data
from math import pi
from Robot_arm import ROBOT

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
robot = ROBOT("anthro_arm_bottle1_demo")
kpt_ee = ROBOT.keypoint(robot, robot.ee_index)
kpt_wrist = ROBOT.keypoint(robot, robot.wrist_index)
kpt_elbow = ROBOT.keypoint(robot, robot.elbow_index)
kpt_shoulder = ROBOT.keypoint(robot, robot.shoulder_index)

joints_indexes = [i for i in range(p.getNumJoints(robot.robot_id)) 
                  if p.getJointInfo(robot.robot_id, i)[2] != p.JOINT_FIXED]

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-135,
                                 cameraPitch=-36, cameraTargetPosition=[0.2,0,0.5])

q = 0.01
joint_idx = 0
while True:
    p.stepSimulation()
    q += 0.01
    # p.resetJointState(bodyUniqueId=robot.robot_id,
    #                   jointIndex=joint_idx,
    #                   targetValue=q,
    #                   targetVelocity=0)
    # keypoint tracking
    p.resetBasePositionAndOrientation(robot.robot_id, [0, -q/10, 1], [0, 0, 0, 1])
    kpt_ee.draw_traj()
    kpt_wrist.draw_traj()
    kpt_elbow.draw_traj()
    kpt_shoulder.draw_traj()
    time.sleep(1./240.)
    eb_position = p.getLinkState(robot.robot_id, robot.elbow_index)[2]
    print(eb_position)

# 断开连接
p.disconnect()