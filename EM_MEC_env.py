# -*- coding: utf-8 -*
import numpy as np
import time
import gym
import math as mt
import matplotlib.pyplot as plt
from time import *
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

B = 2e9          # 系统带宽2G Hz
h0 = 1e-3        # -30dB 参考距离1m处的接收功率
noise = 1e-9     # -60dBm 每个传感器处的噪声功率
p_i = 0.2        # 200mW  每个传感器的传输功率
T_qos = 10       # QoS为10s
phi = 1          # φ
k = 1e-30        # 有效切换电容`
epsilon_low = 0.5    # 最低精度要求
E_max = 5e4      # 最大化总能耗 50KJ
X = 10           # MEC的横坐标
Y = 10           # MEC的纵坐标
D_i = 90         # 传感器到MEC服务器传输的数据量 90mb
t_i = 5
L = 65535 * 8 + 40 * 8


class EM_MEC(object):
    def __init__(self):
        super(EM_MEC, self).__init__()
        self.A_max = 10        # 通信信道的分配向量最大值
        self.F_max = 100       # Haibo: 可以把计算资源认定为100个资源块，每块有1GCPS，不然你这个计算空间太大了。
        self.ESs = 10          # 传感器数量
        self.x_s = 10          # 任意俩相邻单元中心在x轴和y轴的水平坐标的集合
        self.y_s = 10
        self.x_i = 0           # 传感器的横、纵坐标
        self.y_i = 0
        self.l_e = [self.x_i, self.y_i]                           # 传感器位置
        self.l_m = [X, Y]                                         # mec位置处于[X,Y]，传感器分布四周
        self.ac = self.A_max                                      # 剩余信道与资源
        self.fc = self.F_max
        self.N_slot = 1000               # number of time slots in one episode
        self.eps = 50                    # number of episode
        self.a_i = np.zeros((self.eps, self.N_slot), dtype=np.int)        # 优化变量
        self.f_i = np.zeros((self.eps, self.N_slot), dtype=np.int)
        self.state = np.array([0, 0])
        self.h_i = h0 / (mt.pow((X - self.x_i), 2) + mt.pow((Y - self.y_i), 2))   # 传感器传输的信道功率增益
        self.r_i = np.zeros((self.eps, self.N_slot), dtype=np.float)
        self.e = np.zeros((self.eps, self.N_slot), dtype=np.float)                # 总能耗
        self.throughput = np.zeros((self.eps, self.N_slot), dtype=np.float)       # 吞吐量设置
        self.alpha_i = np.zeros((self.eps, self.N_slot), dtype=np.float)          # 计算准确度
        # 穷举算法
        self.r_i_ex = np.zeros((1, self.eps-1), dtype=np.float)
        self.r_i_ex_min = np.zeros((1, self.eps-1), dtype=np.float)
        self.throughput_ex_min = np.zeros((1, self.eps-1), dtype=np.float)
        self.e_ex = np.zeros((1, self.eps-1), dtype=np.float)
        self.e_ex_max = np.zeros((1, self.eps-1), dtype=np.float)
        self.alpha_i_ex = np.zeros((1, self.eps-1), dtype=np.float)
        self.alpha_i_ex_max = np.zeros((1, self.eps-1), dtype=np.float)
        # 贪婪算法
        self.r_i_gr = np.zeros((1, self.eps-1), dtype=np.float)
        self.throughput_gr = np.zeros((1, self.eps-1), dtype=np.float)

        self.n_actions = np.int(self.A_max)*np.int(self.F_max)
        self.n_features = 2

        # generate action table;
        # Haibo: 只需要定义这个action 空间就行了, 利用这个actions 中可以用数组值存放DRL学习得到的action 的index,
        # 然后用行数代表信道数，列数代表计算资源块数。映射出DRL的action的index和 信道数和计算资源块数的取值。参找用矩阵的值，找到具体的横纵坐标的方法。
        self.actions = np.zeros((np.int(self.A_max), np.int(self.F_max)), dtype=np.int)
        index = 0
        for i in range(np.int(self.A_max)):
            for j in range(np.int(self.F_max)):
                self.actions[i, j] = index
                index = index + 1
        print(index)

        # observation
        self.min_ac = 0
        self.max_ac = self.A_max
        self.min_fc = 0
        self.max_fc = self.F_max
        self.low_state = np.array([self.min_ac, self.min_fc])
        self.high_state = np.array([self.max_ac, self.max_fc])
        self._build_em_mec()

    def _build_em_mec(self):
        # 梅：初始化你的终端传感的2D位置、MEC服务器的位置、系统总的通信资源量、系统总的计算资源量，以及你的算法里所有的常量
        self.w_i = np.zeros((self.ESs, 2), dtype=np.float)  # 传感器位置
        # self.Alpha_i = 0.0023 * phi * D_i / self.f_i + beta                         # 情感预测准确性公式

    def reset(self):
        # 梅：算法刚开始的时候有什么变量要重置成初始状态，比如计算和通信资源总量，把优化变量重置成初识变量。
        self.a_i = np.zeros((self.eps, self.N_slot), dtype=np.int)  # 优化变量重置成初识变量
        self.f_i = np.zeros((self.eps, self.N_slot), dtype=np.int)
        self.ac = self.A_max - self.a_i[0, 0]
        self.fc = self.F_max - self.f_i[0, 0]
        self.ac, self.fc = self.state
        return np.array(self.state)  # 返回环境的初始化状态

    def step(self, action, ep, slot):
        # 其输入是动作a，输出是：下一步状态，立即回报，是否终止，调试项。
        self.ac, self.fc = self.state
        self.a_i[ep, slot] = action[0]
        self.f_i[ep, slot] = action[1]

        # 更新信道ai和计算资源fi
        self.out = 0
        if self.a_i[ep, slot] > self.A_max:
            self.a_i[ep, slot] -= self.a_i[ep, slot]
            self.out = 1

        if self.f_i[ep, slot] > self.F_max:
            self.f_i[ep, slot] -= self.f_i[ep, slot]
            self.out = 1

        if self.f_i[ep, slot] < (0.6709 * phi * mt.log(epsilon_low) + 13.28 * phi):
            self.f_i[ep, slot] -= self.f_i[ep, slot]
            self.out = 1

        self.state = np.array([self.ac, self.fc])

        self.alpha_i[ep, slot] = mt.pow(10, ((self.f_i[ep, slot] / phi - 13.28) / 0.6709))

        # 传感器到MEC服务器可实现的上行数据速率
        self.r_i[ep, slot] = np.float(self.a_i[ep, slot]) / self.A_max * B * np.log2(1 + (p_i * self.h_i) / noise)

        self.e[ep, slot] = p_i * t_i + k * phi * self.r_i[ep, slot] * t_i * mt.pow(self.f_i[ep, slot], 2)
        reward = (E_max - self.e[ep, slot]) / E_max
        if self.out == 1:
            reward = reward - 0.1                              # give an additional penality
        self.ac = self.A_max - self.a_i[ep, slot]  # Haibo: update state after all the actions.
        self.fc = self.F_max - self.f_i[ep, slot]

        state_ = np.array([self.ac, self.fc])
        return state_, reward

    def find_action(self, index):
        i = np.int(index / self.F_max)
        j = np.int(index - i*self.F_max)
        action = []
        if index == self.actions[i, j]:
            action.append(i)
            action.append(j)
        return action

    def plot_throughput(self, eps):
        print("Throughput: ")
        # 穷举算法
        begin_time_ex = time()
        for s in range(eps):
            self.r_i_ex_min[0, s] = np.float(1) / self.A_max * B * np.log2(1 + (p_i * self.h_i) / noise)
            for i in range(self.A_max):
                self.r_i_ex[0, s] = np.float(i) / self.A_max * B * np.log2(1 + (p_i * self.h_i) / noise)
                if self.r_i_ex[0, s] > self.r_i_ex_min[0, 0]:
                    self.r_i_ex_min[0, s] = self.r_i_ex[0, s]
            self.throughput_ex_min[0, s] = self.r_i_ex_min[0, s] * t_i / 1e9
        end_time_ex = time()
        run_time_ex = end_time_ex - begin_time_ex
        ave_th_ex = np.sum(self.throughput_ex_min[0, :]) / eps
        print("Average Throughput: exhaustive : %f" % ave_th_ex)
        # print("begin time: exhaustive : %f" % begin_time_ex)
        # print("end time: exhaustive : %f" % end_time_ex)
        print("run time: exhaustive : %f" % run_time_ex)

        # DQN
        begin_time_dqn = time()
        for s2 in range(eps):
            for n2 in range(self.N_slot):
                self.throughput[s2, n2] = self.r_i[s2, n2] * t_i / 1e8
        plot_Th = np.zeros((1, eps), dtype=np.float)
        for e1 in range(eps):
            plot_Th[0, e1] = plot_Th[0, e1] + np.sum(self.throughput[e1, :]) / self.N_slot
        ave_th_dqn = np.sum(plot_Th[0, :]) / eps
        end_time_dqn = time()
        run_time_dqn = end_time_dqn - begin_time_dqn
        print("Average Throughput: DQN : %f" % ave_th_dqn)
        print("run time: DQN : %f" % run_time_dqn)

        # 贪婪算法
        begin_time_greedy = time()
        self.r_i_gr[0, 0] = np.float(self.A_max) / self.A_max * B * np.log2(1 + (p_i * self.h_i) / noise)
        self.throughput_gr[0, 0] = self.r_i_gr[0, 0] * t_i / 1e9
        for e2 in range(eps):
            self.throughput_gr[0, e2] = self.throughput_gr[0, e2] + self.throughput_gr[0, 0]
        ave_th_greedy = np.sum(self.throughput_gr[0, :]) / eps
        end_time_greedy = time()
        run_time_greedy = end_time_greedy - begin_time_greedy
        print("Average Throughput: greedy : %f" % ave_th_greedy)
        # print("begin time: greedy : %f" % begin_time_greedy)
        # print("end time: greedy : %f" % end_time_greedy)
        print("run time: greedy : %f" % run_time_greedy)


        plt.plot(range(eps), self.throughput_ex_min[0, :].T, c='r', linestyle='-', marker='<', label="exhaustive")
        plt.plot(range(eps), plot_Th[0, :].T, c='b', linestyle='-', marker='>', label="DQN")
        plt.plot(range(eps), self.throughput_gr[0, :].T, c='g', linestyle='-', marker='o', label="greedy")
        plt.xlabel('Episode')
        plt.ylabel('Throughput(kbits)')
        plt.legend()
        plt.grid(linestyle='-.')
        plt.show()

    def plot_energy_efficiency(self, eps):
        print("\nEnergy efficiency: ")
        # 穷举算法
        begin_time_ee_ex = time()
        self.e_ex[0, 0] = p_i * t_i + k * phi * self.r_i_ex[0, 0] * t_i * mt.pow(0, 2)
        self.e_ex_max[0, 0] = self.e_ex[0, 0]
        for s in range(eps):
            for i in range(self.A_max):
                self.r_i_ex[0, s] = np.float(i) / self.A_max * B * np.log2(1 + (p_i * self.h_i) / noise)
                for j in range(self.F_max):
                    self.e_ex[0, s] = p_i * t_i + k * phi * self.r_i_ex[0, s] * t_i * mt.pow(j, 2)
                    if self.e_ex[0, s] > self.e_ex_max[0, 0]:
                        self.e_ex_max[0, s] = self.e_ex[0, s]
                    else:
                        self.e_ex_max[0, s] = self.e_ex_max[0, 0]
        plot_ex = np.zeros((1, eps), dtype=np.float)
        for i1 in range(eps):
            plot_ex[0, i1] = self.throughput_ex_min[0, i1] / self.e_ex_max[0, i1] / 1e2
        ave_ex = np.sum(plot_ex[0, :]) / eps
        end_time_ee_ex = time()
        run_time_ee_ex = end_time_ee_ex - begin_time_ee_ex
        print("Energy efficiency : exhaustive : %f" % ave_ex)
        print("run time: exhaustive : %f" % run_time_ee_ex)

        # DQN
        begin_time_ee_dqn = time()
        plot_ee = np.zeros((1, eps), dtype=np.float)
        for i2 in range(eps):
            plot_ee[0, i2] = np.sum(self.throughput[i2, :]) / np.sum(self.e[i2, :]) / 1e-3 /1e5
        ave_dqn = np.sum(plot_ee[0, :]) / eps
        end_time_ee_dqn = time()
        run_time_ee_dqn = end_time_ee_dqn - begin_time_ee_dqn
        print("Energy efficiency : DQN : %f" % ave_dqn)
        print("run time: DQN : %f" % run_time_ee_dqn)

        # 贪婪算法
        begin_time_ee_gr = time()
        r_i_gr1 = np.float(1) / self.A_max * B * np.log2(1 + (p_i * self.h_i) / noise)
        e_gr1 = p_i * t_i + k * phi * r_i_gr1 * t_i * mt.pow(self.F_max, 2)
        throughput_gr1 = r_i_gr1 * t_i / 1e9
        plot_gr = np.zeros((1, eps), dtype=np.float)
        for i4 in range(eps):
            plot_gr[0, i4] = plot_gr[0, i4] + throughput_gr1 / e_gr1 / 10
            i4 = i4 + 1
        ave_gr = np.sum(plot_gr[0, :]) / eps
        end_time_ee_gr = time()
        run_time_ee_gr = end_time_ee_gr - begin_time_ee_gr
        print("Energy efficiency : greedy : %f" % ave_gr)
        print("run time: greedy : %f" % run_time_ee_gr)

        plt.plot(range(eps), plot_ex[0, :].T, c='r', linestyle='-', marker='<', label="exhaustive")
        plt.plot(range(eps), plot_ee[0, :].T, c='b', linestyle='-', marker='>', label="DQN")
        plt.plot(range(eps), plot_gr[0, :].T, c='g', linestyle='-', marker='o', label="greedy")
        plt.xlabel('Episode')
        plt.ylabel('Energy-Efficiency(bits/J)')
        plt.legend()
        plt.grid(linestyle='-.')
        plt.show()

    def plot_Accuracy(self, eps):
        print("\n Accuracy: ")
        # 穷举算法
        begin_time_acc_ex = time()
        self.alpha_i_ex_max[0, 0] = mt.pow(10, ((0 / phi - 13.28) / 0.6709))
        for i in range(eps):
            for j in range(self.F_max):
                self.alpha_i_ex[0, i] = mt.pow(10, ((j / phi - 13.28) / 0.6709))
                if self.alpha_i_ex[0, i] > self.alpha_i_ex_max[0, 0]:
                    self.alpha_i_ex_max[0, i] = self.alpha_i_ex[0, i]
                else:
                    self.alpha_i_ex_max[0, i] = self.alpha_i_ex_max[0, 0]
        plot_ac_ex = np.zeros((1, eps), dtype=np.float)
        for g in range(eps):
            plot_ac_ex[0, g] = plot_ac_ex[0, g] + self.alpha_i_ex_max[0, g] / 1e128
        ave_ac_ex = np.sum(plot_ac_ex[0, :]) / eps
        print("Average Accuracy: exhaustive : %f" % ave_ac_ex)
        end_time_acc_ex = time()
        run_time_acc_ex = end_time_acc_ex - begin_time_acc_ex
        print("run time: exhaustive : %f" % run_time_acc_ex)

        # DQN
        begin_time_acc_dqn = time()
        plot_Ac = np.zeros((1, eps), dtype=np.float)
        for g1 in range(eps):
            plot_Ac[0, g1] = plot_Ac[0, g1] + np.sum(self.alpha_i[g1, :]) / self.N_slot / 1e125
        ave_ac = np.sum(plot_Ac[0, :]) / eps
        print("Average Accuracy: DQN : %f" % ave_ac)
        end_time_acc_dqn = time()
        # print("end time: ex : %f" % end_time_acc_ex)
        # print("begin time: dqn : %f" % begin_time_acc_dqn)
        # print("end time: dqn : %f" % end_time_acc_dqn)
        run_time_acc_dqn = end_time_acc_dqn - begin_time_acc_dqn
        print("run time: dqn : %f" % run_time_acc_dqn)

        # 贪婪算法
        alpha_i_gr = mt.pow(10, ((self.F_max / phi - 13.28) / 0.6709)) / 1e130
        plot_ac_gr = np.zeros((1, eps), dtype=np.float)
        for g2 in range(eps):
            plot_Ac[0, g2] = plot_Ac[0, g2] + alpha_i_gr
        print("Average Accuracy: greedy : %f" % alpha_i_gr)
        end_time_acc_gr = time()
        run_time_acc_gr = end_time_acc_gr - end_time_acc_dqn
        print("run time: greedy : %f" % run_time_acc_gr)

        plt.plot(range(eps), plot_ac_ex[0, :].T, c='r', linestyle='-', marker='<', label="exhaustive")
        plt.plot(range(eps), plot_Ac[0, :].T, c='b', linestyle='-', marker='>', label="DQN")
        plt.plot(range(eps), plot_ac_gr[0, :].T, c='g', linestyle='-', marker='o', label="greedy")
        plt.ylim(0.0, 1.0)
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(linestyle='-.')
        plt.show()





