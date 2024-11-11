import pybullet as p
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from utils import *
from math import pi


class ROBOT:
    def __init__(self, arm, tool, dof=7, 
                 kpt_weight_opt=[5, 5, 10, 1],  # [eb, wr, ee, ee_ori]
                 kpt_weight_PDIK=[2, 1, 10, 1]):
        self.startPos = [0, 0, 1]
        self.startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.dof = dof
        self.init_joint_angles = [0. for i in range(self.dof)]
        self.name = arm+"_"+tool if tool else arm
        self.robot_id = p.loadURDF("models/"+arm+"/"+self.name+"/"+"/urdf/"+self.name+".urdf", 
                      self.startPos, self.startOrientation, useFixedBase=1)
        self.base_bias = np.array(self.startPos) + np.array(np.loadtxt("models/"+arm+"/" + "base_bias.txt"))
        self.joints_indexes = [i for i in range(p.getNumJoints(self.robot_id)) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]
        self.shoulder_index, self.elbow_index, self.wrist_index, self.ee_index = (self.joints_indexes[0], self.joints_indexes[2], 
                                                                                  self.joints_indexes[5], self.joints_indexes[6])
        for i in range(len(self.joints_indexes)):
            p.resetJointState(bodyUniqueId=self.robot_id,
                              jointIndex=i,
                              targetValue=self.init_joint_angles[i],
                              targetVelocity=0)
        self.q_init, self.dq_init, self.ddq_init = self.get_joints_states()
        self.q_01 = np.array(self.q_init[:3])
        self.q_12 = np.array(self.q_init[3])
        self.q_23 = np.array(self.q_init[4:])
        self.kpt_weight_opt = kpt_weight_opt
        self.kpt_weight_PDIK = kpt_weight_PDIK
        self.bound = []
        with open("models/"+arm+"/"+'bounds.txt', 'r') as f:
            for line in f:
                lower, upper = line.strip().split(',')  # 按逗号分隔每一行
                # 处理 None 值
                lower = float(lower) if lower != 'None' else None
                upper = float(upper) if upper != 'None' else None
                self.bound.append((lower, upper))

    def get_joints_states(self):
        joint_states = p.getJointStates(self.robot_id, range(p.getNumJoints(self.robot_id)))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques
    
    def get_error(self, goal_pos, index):
        current_pos = p.getLinkState(self.robot_id, index)[0]
        error = goal_pos - current_pos
        return error
    
    def get_ee_ori_error(self, goal_ori, index):
        current_ori = p.getLinkState(self.robot_id, index)[1]
        # error = R.from_quat(goal_ori).as_matrix() - R.from_quat(current_ori).as_matrix()  # 用旋转矩阵求误差
        error = R.from_quat(goal_ori).as_rotvec() - R.from_quat(current_ori).as_rotvec()  # 用旋转向量求误差
        # error = goal_ori - current_ori
        return error
    
    def FK(self, q):
        for i in range(len(self.joints_indexes)):
            p.resetJointState(bodyUniqueId=self.robot_id,
                              jointIndex=i,
                              targetValue=q[i],
                              targetVelocity=0)
        self.q, self.dq, self.ddq = self.get_joints_states()

    def compute_jacobians(self):
        self.J_01, _ = self.get_jacobian(index=self.elbow_index)
        self.J_01 = self.J_01[:, :3]  # J-01的伪逆要截取一部分，以肘关节为止
        self.J_02, _ = self.get_jacobian(index=self.wrist_index)
        self.J_02 = self.J_02[:, :4]  # J-02的伪逆要截取一部分，以腕关节为止
        self.J_v, self.J_w = self.get_jacobian(index=self.ee_index)

    def get_jacobian(self, index=6):
        self.q, self.dq, self.ddq = self.get_joints_states()
        zero_vec = [0.0] * p.getNumJoints(self.robot_id)
        jac_t, jac_r = p.calculateJacobian(self.robot_id, index, p.getLinkState(self.robot_id, index)[2], self.q, self.dq, zero_vec)
        return np.array(jac_t), np.array(jac_r)
    
    def step_PDIK(self, x_eb, x_wr, x_ee, q_ee, dx_eb, dx_wr, dx_ee, dq_ee, dt=0.01):
        """
        Partial Differential Inverse Kinematics (PDIK)
        """
        ## Compute errors
        # position
        self.eb_error = x_eb - p.getLinkState(self.robot_id, self.elbow_index)[0]
        self.wr_error = x_wr - p.getLinkState(self.robot_id, self.wrist_index)[0]
        self.ee_error = x_ee - p.getLinkState(self.robot_id, self.ee_index)[0]
        # ee orientation
        self.ee_ori = p.getLinkState(self.robot_id, self.ee_index)[1]  # current orientation
        self.ee_error_ori = R.from_quat(q_ee) * R.from_quat(self.ee_ori).inv()
        self.ee_error_ori = self.ee_error_ori.as_rotvec()
        # overall error
        self.error_all = [np.linalg.norm(self.eb_error, ord=2), np.linalg.norm(self.wr_error, ord=2), 
                     np.linalg.norm(self.ee_error, ord=2), np.linalg.norm(R.from_quat(q_ee).as_matrix() - R.from_quat(self.ee_ori).as_matrix(), ord=2)]
        # print(error_all, np.sum(error_all))

        ## Compute dq
        self.dq_01_eb = self.DLS(self.J_01).dot(dx_eb + self.kpt_weight_PDIK[0] * self.eb_error)  # elbow effect
        self.dq_wr = self.DLS(self.J_02).dot(dx_wr + self.kpt_weight_PDIK[1] * self.wr_error)  # wrist effect
        self.dq_ee_v = self.DLS(self.J_v).dot(dx_ee + self.kpt_weight_PDIK[2] * self.ee_error)  # ee position effect
        self.dq_ee_w = self.DLS(self.J_w).dot(dq_ee + self.kpt_weight_PDIK[3] * self.ee_error_ori)  # ee orientation effect

        self.dq_01_wr, self.dq_12_wr = self.dq_wr[:3], self.dq_wr[3]    
        self.dq_01_ee_v, self.dq_12_ee_v, self.dq_23_ee_v = self.dq_ee_v[:3], self.dq_ee_v[3], self.dq_ee_v[4:]
        self.dq_01_ee_w, self.dq_12_ee_w, self.dq_23_ee_w = self.dq_ee_w[:3], self.dq_ee_w[3], self.dq_ee_w[4:]

        ## Compute q
        self.q_01 += (self.dq_01_eb + self.dq_01_wr + self.dq_01_ee_v + self.dq_01_ee_w) * dt
        self.q_12 += (self.dq_12_wr + self.dq_12_ee_v + self.dq_12_ee_w) * dt
        self.q_23 += (self.dq_23_ee_v + self.dq_23_ee_w) * dt
        q = np.hstack((self.q_01, self.q_12, self.q_23))
        self.FK(q)
    
    def DLS(self, J, min_damping=0.1, max_damping=0.1):
        """
        使用自适应阻尼最小二乘法计算广义逆雅可比。
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

    def kpt_opt(self, sample_len, elbow_traj, wrist_traj, ee_traj, ee_ori=0):
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
                Error.append(pos_error * self.kpt_weight_opt[2])
                Error.append(ori_error * self.kpt_weight_opt[3])
                i += self.dof
            for g in wrist_traj:
                self.FK(q[j:j+self.dof])
                pos_error = np.linalg.norm(self.get_error(g, self.wrist_index), ord=2)
                Error.append(pos_error * self.kpt_weight_opt[1])
                j += self.dof
            for g in elbow_traj:
                self.FK(q[k:k+self.dof])
                pos_error = np.linalg.norm(self.get_error(g, self.elbow_index), ord=2)
                Error.append(pos_error * self.kpt_weight_opt[0])
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
    
    def step_kpt_opt(self, x_eb, x_wr, x_ee, q_ee, q_init):
        """
        差分位置优化控制
        """
        q_init = q_init  # 长度为n×m（目标数×关节自由度数）
        def eqn(q):
            Error = []
            self.FK(q)
            # elbow
            pos_error = np.linalg.norm(self.get_error(x_eb, self.elbow_index), ord=2)
            Error.append(pos_error * self.kpt_weight_opt[0])

            # wrist
            pos_error = np.linalg.norm(self.get_error(x_wr, self.wrist_index), ord=2)
            Error.append(pos_error * self.kpt_weight_opt[1])

            # ee
            pos_error = np.linalg.norm(self.get_error(x_ee, self.ee_index), ord=2)
            ori_error = np.linalg.norm(self.get_ee_ori_error(q_ee, self.ee_index), ord=2)
            Error.append(pos_error * self.kpt_weight_opt[2])
            Error.append(ori_error * self.kpt_weight_opt[3])

            # 以下几种误差的求法都可以，QP最好
            # Error = np.linalg.norm(np.array(Error))
            # Error = np.sum(np.array(Error))
            Error = np.dot(np.array(Error).T, np.array(Error)) + 0.001 *np.dot(q.T, q)  # QP问题
            return Error
        q_star = minimize(eqn, q_init, method='BFGS')
        Error = q_star.fun
        q_star = np.array(q_star.x)
        return q_star
    

    def feature_space_opt_position(self, pca, kpt_list, mu, cons_dict, ee_ori, q_init, q_base2tg, t_base2tg, cons_opt=True):
        """
        假设ee方向为确定约束，仅针对关键点位置进行子空间优化

        Parameters
        ----------
        pca : pca model
        kpt_list : a list containing the index of kpts (e.g. [robot.elbow_index, robot.wrist_index])（PCA梯度×机器人雅可比）
        cons_dict : a dict containing explicit constrains ({index: cons_t_base2k})
        ee_ori : 末端朝向
        q_base2tg, t_base2tg : 目标点位在机器人坐标系下的位姿

        Returns
        ----------

        """
        q_init = q_init  # 长度为n×m（目标数×关节自由度数）
        bounds = self.bound
        def func(q):
            self.FK(q)
            x = []  # keypoint positions
            for index in kpt_list:
                t_base2k = p.getLinkState(self.robot_id, index)[0]
                q_base2k = np.array([0, 0, 0, 1])  # 不关心关键点的朝向
                # 这里需要先把笛卡尔坐标转换到target坐标下
                _, t_tg2k, _ = coordinate_transform(q_base2k, t_base2k, q_base2tg, t_base2tg)
                x = np.hstack((x, t_tg2k))
            x = x.reshape((1, -1))
            z = pca.transform(x)
            E = np.dot(z - mu, (z - mu).T)  # subspace error
            E = E.reshape(-1)
            return E
        
        def func_jac(q):
            self.FK(q)
            x = []  # keypoint positions
            J_kpt = np.empty((0, 7))  # keypoint jacobians
            for index in kpt_list:
                ### 关键点位置x
                t_base2k = p.getLinkState(self.robot_id, index)[0]
                q_base2k = np.array([0, 0, 0, 1])  # 不关心关键点的朝向
                # 这里需要先把笛卡尔坐标转换到target坐标下
                _, t_tg2k, _ = coordinate_transform(q_base2k, t_base2k, q_base2tg, t_base2tg)
                x = np.hstack((x, t_tg2k))
                ### 雅可比矩阵
                J_v, _ = self.get_jacobian(index)
                J_kpt = np.vstack((J_kpt, J_v))
            x = x.reshape((1, -1))
            z = pca.transform(x)  # shape (1, 3)
            jac = 2*(z - mu) @ pca.components_ @ J_kpt  # (1, 3), (3, 6), (6, 7)
            return jac
        
        def cons_position(q):
            self.FK(q)
            index = list(cons_dict.keys())[0]
            mu_cons = cons_dict[index]  # 约束均值（笛卡尔空间）
            e = self.get_error(mu_cons, index)
            return - np.dot(e.T, e) + 0.000225
        
        def cons_position_jac(q):
            self.FK(q)
            index = list(cons_dict.keys())[0]
            mu_cons = cons_dict[index]  # 约束均值（笛卡尔空间）
            e = self.get_error(mu_cons, index)
            J_v, _ = self.get_jacobian(index)
            jac = 2 * e @ J_v
            return jac
        
        def cons_orientation(q):
            self.FK(q)
            index = self.ee_index
            e = self.get_ee_ori_error(ee_ori, index)
            return - np.dot(e.T, e) + 0.015
        
        def cons_orientation_jac(q):
            self.FK(q)
            index = self.ee_index
            e = self.get_ee_ori_error(ee_ori, index)
            _, J_w = self.get_jacobian(index)
            jac = 2 * e @ J_w
            return jac
        
        cons = [{'type': 'ineq', 
                 'fun': cons_position}, 
                {'type': 'ineq', 
                 'fun': cons_orientation}]
        
        if cons_opt:
            # 约束优化
            q_star = minimize(func, q_init, method='COBYLA', jac=func_jac, constraints=cons, bounds=bounds)
        else:
            # 无约束优化
            def eqn(q):
                return func(q) - 50*cons_position(q) - cons_orientation(q)
            q_star = minimize(eqn, q_init, method='BFGS')

        Error = q_star.fun
        q_star = np.array(q_star.x)
        return q_star
    
    def cartesian_space_opt_position(self, kpt_list, mu, cons_dict, ee_ori, q_init, q_base2tg, t_base2tg, cons_opt=True):
        """
        假设ee方向为确定约束，仅针对关键点位置进行笛卡尔空间优化

        Parameters
        ----------
        kpt_list : a list containing the index of kpts (e.g. [robot.elbow_index, robot.wrist_index])
        cons_dict : a dict containing explicit constrains ({index: cons_t_base2k})
        ee_ori : 末端朝向
        q_base2tg, t_base2tg : 目标点位在机器人坐标系下的位姿

        Returns
        ----------

        """
        q_init = q_init  # 长度为n×m（目标数×关节自由度数）
        bounds = self.bound
        print(bounds)
        def func(q):
            self.FK(q)
            x = []  # keypoint positions
            for index in kpt_list:
                t_base2k = p.getLinkState(self.robot_id, index)[0]
                q_base2k = np.array([0, 0, 0, 1])  # 不关心关键点的朝向
                # 这里需要先把笛卡尔坐标转换到target坐标下
                _, t_tg2k, _ = coordinate_transform(q_base2k, t_base2k, q_base2tg, t_base2tg)
                x = np.hstack((x, t_tg2k))
            x = x.reshape(-1)
            E = np.dot(x - mu, (x - mu).T)  # subspace error
            return E
        
        def func_jac(q):
            self.FK(q)
            x = []  # keypoint positions
            J_kpt = np.empty((0, 7))  # keypoint jacobians
            for index in kpt_list:
                ### 关键点位置x
                t_base2k = p.getLinkState(self.robot_id, index)[0]
                q_base2k = np.array([0, 0, 0, 1])  # 不关心关键点的朝向
                # 这里需要先把笛卡尔坐标转换到target坐标下
                _, t_tg2k, _ = coordinate_transform(q_base2k, t_base2k, q_base2tg, t_base2tg)
                x = np.hstack((x, t_tg2k))
                ### 雅可比矩阵
                J_v, _ = self.get_jacobian(index)
                J_kpt = np.vstack((J_kpt, J_v))
            x = x.reshape((1, -1))
            jac = 2*(x - mu) @ J_kpt  # (1, 3), (3, 6), (6, 7)
            return jac
        
        def cons_position(q):
            self.FK(q)
            index = list(cons_dict.keys())[0]
            mu_cons = cons_dict[index]  # 约束均值（笛卡尔空间）
            e = self.get_error(mu_cons, index)
            return - np.dot(e.T, e) + 0.000225
        
        def cons_position_jac(q):
            self.FK(q)
            index = list(cons_dict.keys())[0]
            mu_cons = cons_dict[index]  # 约束均值（笛卡尔空间）
            e = self.get_error(mu_cons, index)
            J_v, _ = self.get_jacobian(index)
            jac = 2 * e @ J_v
            return jac
        
        def cons_orientation(q):
            self.FK(q)
            index = self.ee_index
            e = self.get_ee_ori_error(ee_ori, index)
            return - np.dot(e.T, e) + 0.015
        
        def cons_orientation_jac(q):
            self.FK(q)
            index = self.ee_index
            e = self.get_ee_ori_error(ee_ori, index)
            _, J_w = self.get_jacobian(index)
            jac = 2 * e @ J_w
            return jac
        
        cons = [{'type': 'ineq', 
                 'fun': cons_position}, 
                {'type': 'ineq', 
                 'fun': cons_orientation}]
        
        if cons_opt:
            # 约束优化
            q_star = minimize(func, q_init, method='COBYLA', jac=func_jac, constraints=cons, bounds=bounds)
        else:
            # 无约束优化
            def eqn(q):
                return func(q) - 50*cons_position(q) - cons_orientation(q)  # 50在这里只是随便设的权重，理念是方差越小的p2p约束，这个权重应该越大
            q_star = minimize(eqn, q_init, method='BFGS')

        Error = q_star.fun
        q_star = np.array(q_star.x)
        return q_star



            

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