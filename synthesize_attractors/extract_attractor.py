import pybullet as p
import time
import pybullet_data
import math
import os, sys
sys.path.append(os.getcwd())
from utils import *
from Robot_arm import ROBOT


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # 先不渲染
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setGravity(0,0,0)
planeId = p.loadURDF("plane.urdf")
robot = ROBOT("arm_bottle1_demo")
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

file_list = [i for i in range(0,54)]
start_attractor = []
end_attractor = []
for file_index in file_list:
    file_name = file_path + files[file_index]
    segment_index = int(segment_file[file_index])
    # start attractor
    _, start_base2eb, _, start_base2wr, start_q_base2ee, start_base2ee = get_transformed_trajectory(file_name, 
                                                                  base_position,
                                                                  cut_data=[segment_index, segment_index+1],
                                                                  orientation=True)
    sample_len = len(start_base2ee)
    p.addUserDebugPoints(start_base2ee, [([1, 0, 0]) for i in range(sample_len)], 5)
    p.addUserDebugPoints(start_base2wr, [([1, 0, 0]) for i in range(sample_len)], 5)
    p.addUserDebugPoints(start_base2eb, [([1, 0, 0]) for i in range(sample_len)], 5)
    start_attractor.append(np.hstack((start_base2eb, start_base2wr, start_base2ee, start_q_base2ee)).reshape(-1).tolist())
    # end attractor
    _, end_base2eb, _, end_base2wr, end_q_base2ee, end_base2ee = get_transformed_trajectory(file_name, 
                                                                  base_position,
                                                                  cut_data=[-2, -1],
                                                                  orientation=True)
    p.addUserDebugPoints(end_base2ee, [([0, 0, 1]) for i in range(sample_len)], 5)
    p.addUserDebugPoints(end_base2wr, [([0, 0, 1]) for i in range(sample_len)], 5)
    p.addUserDebugPoints(end_base2eb, [([0, 0, 1]) for i in range(sample_len)], 5)
    end_attractor.append(np.hstack((end_base2eb, end_base2wr, end_base2ee, end_q_base2ee)).reshape(-1).tolist())

np.savetxt(main_path + "start_attractor.txt", start_attractor)
np.savetxt(main_path + "end_attractor.txt", end_attractor)

while True:
    p.stepSimulation()
    time.sleep(1./240.)
    
# # 断开连接
# p.disconnect()
