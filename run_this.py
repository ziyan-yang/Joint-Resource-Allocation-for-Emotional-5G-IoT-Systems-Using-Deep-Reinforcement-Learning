# -*- coding: utf-8 -*
from EM_MEC_env import EM_MEC
from RL_brain import DeepQNetwork
import numpy as np

if __name__ == "__main__":
    env = EM_MEC()
    RL = DeepQNetwork(env.n_actions,
                      env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,                         
                      # output_graph=True
                      )
    Episodes = env.eps                                    # self.eps = 80
    ES_dqn = np.zeros((Episodes, env.N_slot, 2), dtype=np.float)

    for ep in range(Episodes):
        observation = env.reset()                         # initial observation
        for slot in range(env.N_slot):                    # self.N_slot = 1000
            action_index = RL.choose_action(observation)  # RL choose action based on observation
            action = env.find_action(action_index)
            observation_, reward = env.step(action, ep, slot)      # RL take action and get next observation and reward
            RL.store_transition(observation, action, reward, observation_)
            ES_dqn[ep, slot, :] = observation_[:]
            if env.N_slot*ep + slot > RL.memory_size:
                RL.learn()
            observation = observation_  # swap observation

        # Haibo: 接下来根据这个action index, 去查表self.actions,
        # 找到这个action index 对应的横纵坐标，代表信道分配和计算资源块分配。见em_mec_env, 51,52行。
        # em_emc_env中写一个find_action 函数，参考我无人机DRL的代码。
        # 保证你下面这个 env.step(action)里面的action 是[x,y]这样一个数组对。
    #RL.plot_Q_value()
    RL.plot_cost()

    EPS = env.eps-1
    env.plot_throughput(EPS)
    env.plot_energy_efficiency(EPS)
    env.plot_Accuracy(EPS)


