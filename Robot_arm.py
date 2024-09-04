import pybullet as p
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R


class ROBOT:
    def __init__(self, name, dof=7, kpt_weight=[10, 5, 10, 1]):
        self.startPos = [0, 0, 1]
        self.startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.dof = dof
        self.init_joint_angles = [0. for i in range(self.dof)]
        self.robot_id = p.loadURDF("models/"+name+"/urdf/"+name+".urdf", 
                      self.startPos, self.startOrientation, useFixedBase=1)
        self.joints_indexes = [i for i in range(p.getNumJoints(self.robot_id)) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]
        self.shoulder_index, self.elbow_index, self.wrist_index, self.ee_index = (self.joints_indexes[0], self.joints_indexes[2], 
                                                                                  self.joints_indexes[5], self.joints_indexes[6])
        for i in range(len(self.joints_indexes)):
            p.resetJointState(bodyUniqueId=self.robot_id,
                              jointIndex=i,
                              targetValue=self.init_joint_angles[i],
                              targetVelocity=0)
        self.q_init, self.dq_init, self.ddq_init = self.get_joints_states()
        self.kpt_weight = kpt_weight

    def get_joints_states(self):
        joint_states = p.getJointStates(self.robot_id, range(p.getNumJoints(self.robot_id)))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques
    
    def get_error(self, goal_pos, index, ee=True):
        current_pos = p.getLinkState(self.robot_id, index)[0]
        error = goal_pos - current_pos
        return error
    
    def get_ee_ori_error(self, goal_ori, index, ee=True):
        current_ori = p.getLinkState(self.robot_id, index)[1]
        error = R.from_quat(goal_ori).as_matrix() - R.from_quat(current_ori).as_matrix()
        # error = goal_ori - current_ori
        return error
    
    def FK(self, q):
        for i in range(len(self.joints_indexes)):
            p.resetJointState(bodyUniqueId=self.robot_id,
                              jointIndex=i,
                              targetValue=q[i],
                              targetVelocity=0)
        self.q, self.dq, self.ddq = self.get_joints_states()

    def get_jacobian(self, index=6):
        self.q, self.dq, self.ddq = self.get_joints_states()
        zero_vec = [0.0] * p.getNumJoints(self.robot_id)
        jac_t, jac_r = p.calculateJacobian(self.robot_id, index, p.getLinkState(self.robot_id, index)[2], self.q, self.dq, zero_vec)
        return np.array(jac_t), np.array(jac_r)
    
    def DLS(self, J, min_damping=0.1, max_damping=0.1):
        """
        使用自适应阻尼最小二乘法计算关节速度。
        参数:
        J (ndarray): 雅可比矩阵，维度为 (m, n)。
        min_damping (float): 最小阻尼因子。
        max_damping (float): 最大阻尼因子。
        """
        # 计算奇异值分解
        U, S, Vt = np.linalg.svd(J)
        damping_factor = np.clip(min_damping / (S.min() + 1e-10), min_damping, max_damping)
        # 计算 J^T (J J^T + λ^2 I)^-1
        m, n = J.shape
        identity_matrix = np.eye(m)
        JJT = np.dot(J, J.T)
        damped_matrix = JJT + (damping_factor ** 2) * identity_matrix
        damped_inverse = np.linalg.inv(damped_matrix)
        J_DLS = np.dot(J.T, damped_inverse)
        return J_DLS

    def opt_kpt(self, sample_len, elbow_traj, wrist_traj, ee_traj, ee_ori=0):
        """
        考虑全局最优
        """
        q_init = [0. for i in range(sample_len*self.dof)]  # 长度为n×m（目标数×关节自由度数）
        def eqn(q):
            Error = []
            i, j, k= 0, 0, 0
            for g, o in zip(ee_traj,ee_ori):
                self.FK(q[i:i+self.dof])
                pos_error = np.linalg.norm(self.get_error(g, self.ee_index), ord=2)
                ori_error = np.linalg.norm(self.get_ee_ori_error(o, self.ee_index), ord=2)
                Error.append(pos_error * self.kpt_weight[2])
                Error.append(ori_error * self.kpt_weight[3])
                i += self.dof
            for g in wrist_traj:
                self.FK(q[j:j+self.dof])
                pos_error = np.linalg.norm(self.get_error(g, self.wrist_index), ord=2)
                Error.append(pos_error * self.kpt_weight[1])
                j += self.dof
            for g in elbow_traj:
                self.FK(q[k:k+self.dof])
                pos_error = np.linalg.norm(self.get_error(g, self.elbow_index), ord=2)
                Error.append(pos_error * self.kpt_weight[0])
                k += self.dof
            # 以下几种误差的求法都可以，QP最好
            # Error = np.linalg.norm(np.array(Error))
            # Error = np.sum(np.array(Error))
            Error = np.dot(np.array(Error).T, np.array(Error)) + 0.001 *np.dot(q.T, q)  # QP问题
            return Error
        Q_star = minimize(eqn, q_init, method='BFGS')
        Error = Q_star.fun
        Q_star = np.array(Q_star.x).reshape([sample_len,self.dof])
        return Q_star, Error

    class keypoint:
        def __init__(self, robot, index, hasPrevPose=0, prevPose=0):
            self.robot = robot
            self.robot_id = robot.robot_id
            self.index = index
            self.hasPrevPose = hasPrevPose
            self.prevPose = prevPose
            self.traj = []
            if self.index == self.robot.ee_index:
                self.color = [1, 0, 0]
            elif self.index == self.robot.wrist_index:
                self.color = [0, 1, 0]
            else:
                self.color = [0, 0, 1]

        def save_traj(self):
            ls = p.getLinkState(self.robot_id, self.index)
            if self.index == self.robot.ee_index:
                self.traj.append(np.hstack((ls[0], ls[1])))
            else:
                self.traj.append(ls[0])
            
        def draw_traj(self):
            ls = p.getLinkState(self.robot_id, self.index)
            if (self.hasPrevPose):
                p.addUserDebugLine(self.prevPose, ls[0], self.color, 3, 15)
            self.hasPrevPose = 1
            self.prevPose = ls[0]

        def reset_pose(self):
            self.hasPrevPose = 0
            self.prevPose = 0