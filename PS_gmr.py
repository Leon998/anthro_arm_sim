import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from gmr import GMM, kmeansplusplus_initialization, covariance_initialization
import pybullet as p
import time
import pybullet_data
from utils import *
from Robot_arm import ROBOT


def GMR_sample(X_train, target_position):
    # GMR
    random_state = np.random.RandomState(0)
    n_components = 5
    initial_means = kmeansplusplus_initialization(X_train, n_components, random_state)
    initial_covs = covariance_initialization(X_train, n_components)
    bgmm = BayesianGaussianMixture(
        n_components=n_components, max_iter=500,
        random_state=random_state).fit(X_train)
    gmm = GMM(n_components=n_components, priors=bgmm.weights_, means=bgmm.means_,
              covariances=bgmm.covariances_, random_state=random_state)
    cgmm = gmm.condition([6, 7, 8], target_position)  # [6, 7, 8]是ee的位置，近似代替target_position
    sampled_position = cgmm.to_mvn().mean
    return sampled_position


main_path = 'trajectories/mocap_csv/710/bottle/'
start_attractor = np.loadtxt(main_path + "start_attractor.txt")  # eb, wr, ee
end_attractor = np.loadtxt(main_path + "end_attractor.txt")  # eb, wr, ee

Attractor = end_attractor  # change this into start or end
X_train = Attractor[::3]
index = 29
test_attractor = Attractor[index]
target_position = test_attractor[6:9]
print(target_position)
sampled_position = GMR_sample(X_train, target_position)
print(sampled_position)
print(test_attractor[0:6])


# # simulation
# physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
# p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # 先不渲染
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# p.setGravity(0,0,0)
# planeId = p.loadURDF("plane.urdf")
# robot = ROBOT("arm_bottle1_demo")
# kpt_ee = ROBOT.keypoint(robot, robot.ee_index)

# # Rendering
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
# p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-135,
#                                  cameraPitch=-36, cameraTargetPosition=[0.2,0,0.5])

# base_position = np.array(robot.startPos) + np.array([-0.05, 0.1, -0.15])  # 肩宽、肩厚、肩高补偿

# ts_base2ee = target_position.reshape(1,3)
# ts_base2wr = sampled_position[3:6].reshape(1,3)
# ts_base2eb = sampled_position[0:3].reshape(1,3)
# sample_len = len(ts_base2ee)
# Q_star, Error = robot.kpt_opt(sample_len, ts_base2ee, ts_base2wr, ts_base2eb)

# loop = False
# while True:
#     p.stepSimulation()
#     time.sleep(1./240.)
#     if loop:
#         robot.FK(robot.init_joint_angles)
#         time.sleep(0.5)
#     for q_star in Q_star:
#         print("q_star: ", q_star)
#         robot.FK(q_star)
#         time.sleep(0.25)