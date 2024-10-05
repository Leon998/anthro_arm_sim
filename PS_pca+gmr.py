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

file_list = [i for i in range(0,54)]
goals = []
for file_index in file_list:
    file_name = file_path + files[file_index]
    segment_index = int(segment_file[file_index])
    # goal attractor
    goal_tg2eb, goal_tg2wr, goal_tg2ee = get_transformed_trajectory(file_name, 
                                                                  base_position,
                                                                  cut_data=[-2, -1],
                                                                  tg_based=True)
    if file_index < 27:
        p.addUserDebugPoints(goal_tg2wr, [[0, 1, 0]], 5)
        p.addUserDebugPoints(goal_tg2eb, [[0, 0, 1]], 5)
    else:
        p.addUserDebugPoints(goal_tg2wr, [[0.5, 1, 0.5]], 5)
        p.addUserDebugPoints(goal_tg2eb, [[0.5, 0.5, 1]], 5)
    # TODO 需要先对所有关键点做PCA，方差小于某个阈值，那么说明是强约束，否则拿去做GMM

    ######################################################################
    goals.append(np.hstack((goal_tg2eb, goal_tg2wr)).reshape(-1))

# PCA on eb and wr
goals = np.array(goals)
print("Original goal shape: ", goals.shape)
pca = decomposition.PCA(n_components=3)
pca.fit(goals)
X = pca.transform(goals)
print("Transformed component shape: ", X.shape)
print("Explained variance ratio: ", pca.explained_variance_ratio_)

# GMR
X_train = np.hstack((X, target_positions))
print("GMR train set shape: ", X_train.shape)

# 测试时的目标位置
test_index = 29
target_position = target_positions[test_index]
sampled_position = GMR_sample(X_train, target_position)
print("Sampled component mean: ", sampled_position)


### TODO Optimization in pca space ###



######################################


# 直接逆变换到笛卡尔空间得到关键点位置，老方法。
sampled_goal = pca.inverse_transform(sampled_position.reshape(1, -1)).reshape(-1)
goal_tg2eb, goal_tg2wr = sampled_goal[:3], sampled_goal[3:]
print(goal_tg2eb, goal_tg2wr)

# visualizing manifold
fig = plt.figure(1, figsize=(4, 3))
ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
ax.scatter(X[:27, 0], X[:27, 1], X[:27, 2], c='g')
ax.scatter(X[27:, 0], X[27:, 1], X[27:, 2], c='b')
plt.show()


while True:
    p.stepSimulation()
    time.sleep(1./240.)