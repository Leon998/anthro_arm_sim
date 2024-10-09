import pybullet as p
import time
import pybullet_data
import math
import os, sys
sys.path.append(os.getcwd())
from utils import *
from Robot_arm import ROBOT
from sklearn import decomposition
from sklearn.mixture import BayesianGaussianMixture
from gmr import GMM, kmeansplusplus_initialization, covariance_initialization
import matplotlib.pyplot as plt


def GMR_sample(X_train, target_position):
    """
    Parameters
    ----------
    X_train : array-like of shape (n_samples, 6) (6 = 3 for pca dim + 3 for target_positions)
    target_position : conditioned target position, shape (1, 3)

    Returns
    ----------
    sampled_position : mean value of GMM under the conditioned target_position, shape (1, 3)
    """
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
    cgmm = gmm.condition([-3, -2, -1], target_position)
    sampled_position = cgmm.to_mvn().mean
    return sampled_position


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
attractor = np.loadtxt(main_path + "end_attractor.txt")  # eb, wr, ee
target_positions = attractor[:, 6:9]
target_orientations = attractor[:, 9:]

file_list = [i for i in range(0,54)]
# keypoint docker
Goal_tg2eb = np.empty((0, 3))
Goal_tg2wr = np.empty((0, 3))
Goal_tg2ee = np.empty((0, 3))
Goal_q_tg2ee = np.empty((0, 3))
for file_index in file_list:
    file_name = file_path + files[file_index]
    segment_index = int(segment_file[file_index])
    # goal attractor
    _, goal_tg2eb, _, goal_tg2wr, goal_q_tg2ee, goal_tg2ee = get_transformed_trajectory(file_name, 
                                                                  base_position,
                                                                  cut_data=[-2, -1],
                                                                  orientation=True,
                                                                  tg_based=True)
    if file_index < 27:
        p.addUserDebugPoints(goal_tg2ee, [[1, 0, 0]], 5)
        p.addUserDebugPoints(goal_tg2wr, [[0, 1, 0]], 5)
        p.addUserDebugPoints(goal_tg2eb, [[0, 0, 1]], 5)
    else:
        p.addUserDebugPoints(goal_tg2ee, [[1, 0.5, 0.5]], 5)
        p.addUserDebugPoints(goal_tg2wr, [[0.5, 1, 0.5]], 5)
        p.addUserDebugPoints(goal_tg2eb, [[0.5, 0.5, 1]], 5)
    # 先实现四元数到欧氏空间转换
    log_q = quaternion2euler(goal_q_tg2ee.reshape(-1))
    ###############################
    Goal_tg2eb = np.vstack((Goal_tg2eb, goal_tg2eb))
    Goal_tg2wr = np.vstack((Goal_tg2wr, goal_tg2wr))
    Goal_tg2ee = np.vstack((Goal_tg2ee, goal_tg2ee))
    Goal_q_tg2ee = np.vstack((Goal_q_tg2ee, log_q))

# 需要先对所有关键点做PCA，方差小于某个阈值，那么说明是强约束，否则拿去做GMM
pca_eb = decomposition.PCA(n_components=3)
pca_wr = decomposition.PCA(n_components=3)
pca_ee = decomposition.PCA(n_components=3)
pca_q_ee = decomposition.PCA(n_components=3)

pca_eb.fit(Goal_tg2eb)
print("Elbow's explained variance: ", pca_eb.explained_variance_)
# print("Elbow's explained variance ratio: ", pca_eb.explained_variance_ratio_)
pca_wr.fit(Goal_tg2wr)
print("Wrist's explained variance: ", pca_wr.explained_variance_)
# print("Wrist's explained variance ratio: ", pca_wr.explained_variance_ratio_)
pca_ee.fit(Goal_tg2ee)
print("EE's explained variance: ", pca_ee.explained_variance_)
# print("EE's explained variance ratio: ", pca_ee.explained_variance_ratio_)
pca_q_ee.fit(Goal_q_tg2ee)
print("EE orientation's explained variance: ", pca_q_ee.explained_variance_)
# print("EE orientation's explained variance ratio: ", pca_q_ee.explained_variance_ratio_)

# 经PCA后发现，EE为p2p约束，EE orientation在转换到欧氏空间后为p2l约束，因此可以直接采样出约束点
# 现直接定义约束
constrain_goal_tg2ee = pca_ee.mean_
constrain_goal_q_tg2ee = euler2quaternion(pca_q_ee.mean_)
print("Constrains: ", constrain_goal_tg2ee, constrain_goal_q_tg2ee)

###################################################################
# PCA on eb and wr，定义为IC（implicit constrain，隐式约束）
Goal_tg2IC = np.hstack((Goal_tg2eb, Goal_tg2wr))
print("Original goal shape: ", Goal_tg2IC.shape)
pca = decomposition.PCA(n_components=3)
pca.fit(Goal_tg2IC)
X = pca.transform(Goal_tg2IC)
print("Transformed component shape: ", X.shape)
print("Explained variance: ", pca.explained_variance_)
# print("Explained variance ratio: ", pca.explained_variance_ratio_)

# # visualizing manifold
# fig = plt.figure(1, figsize=(4, 3))
# ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
# X = Goal_tg2IC
# ax.scatter(X[:27, 0], X[:27, 1], X[:27, 2], c='g')
# ax.scatter(X[27:, 0], X[27:, 1], X[27:, 2], c='b')
# plt.show()

#################################################################

# GMR
X_train = np.hstack((X, target_positions))
print("GMR train set shape: ", X_train.shape)

# 测试时的目标位置
test_index = 29
t_base2tg = target_positions[test_index]
q_base2tg = target_orientations[test_index]
sampled_position = GMR_sample(X_train, t_base2tg)
print("Sampled component mean: ", sampled_position)


# ### TODO Optimization in pca space ###



# ######################################


# # 直接逆变换到笛卡尔空间得到关键点位置，老方法。
# sampled_goal = pca.inverse_transform(sampled_position.reshape(1, -1)).reshape(-1)
# goal_tg2eb, goal_tg2wr = sampled_goal[:3], sampled_goal[3:]
# print(goal_tg2eb, goal_tg2wr)




while True:
    p.stepSimulation()
    time.sleep(1./240.)