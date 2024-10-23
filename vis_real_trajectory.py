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
robot = ROBOT("arm_bottle2_demo")
kpt_ee = ROBOT.keypoint(robot, robot.ee_index)

# Rendering
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-135,
                                 cameraPitch=-36, cameraTargetPosition=[0.2,0,0.5])

main_path = 'trajectories/mocap_csv/1015/chy/'
tool = 'bottle/'
base_bias = np.loadtxt(main_path + "base_bias.txt")  # 肩宽、肩厚、肩高补偿
base_position = np.array(robot.startPos) + np.array(base_bias)
file_path = main_path + tool
files = os.listdir(file_path)
# segment_file = np.loadtxt(main_path + tool + "segment.txt")

file_index = 15
file_name = file_path + files[file_index]
# segment_index = int(segment_file[file_index])

_, ts_base2eb, _, ts_base2wr, qs_base2ee, ts_base2ee, _, t_base2tg = get_transformed_trajectory(file_name, 
                                                                                  base_position,
                                                                                  cut_data=[0, -1],
                                                                                  orientation=True)
# for t in ts_base2eb:
#     d = math.sqrt(math.pow(t[0],2)+math.pow(t[1],2)+math.pow(t[2],2))
#     print(d)

num_points = len(ts_base2ee)
print(ts_base2ee.shape)
p.addUserDebugPoints(ts_base2ee, [([1, 0, 0]) for i in range(num_points)], 5)
p.addUserDebugPoints(ts_base2wr, [([0, 1, 0]) for i in range(num_points)], 5)
p.addUserDebugPoints(ts_base2eb, [([0, 0, 1]) for i in range(num_points)], 5)

while True:
    p.stepSimulation()
    time.sleep(1./240.)
    
# # 断开连接
# p.disconnect()
