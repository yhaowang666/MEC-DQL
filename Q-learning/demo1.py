# -*- codeing = utf-8 -*-
# @Time : 2020/11/i22 18:33
# @Author : 王浩
# @File : demo1.py
# @Software : PyCharm
import numpy as np
import pandas as pd
# 控制探索者agent移动速度有多块
import time

np.random.seed(2)   # 计算机产生一组伪随机数

# 参数设置
N_STATES = 6  # 总状态数
ACTIONS = ['left', 'right']  # 可能采取的动作
EPSILON = 0.9   # 每次选择最优（回报最大）动作的概率
ALPHA = 0.1  # 学习效率，值越大，Q表更新的幅度越大
LAMBDA = 0.9  # 衰减度，未来奖励的衰减值，值越大，未来的奖励越重要，agent看得越远[0,1]
MAX_EPISODES = 13  # 总的回合次数
FRESH_TIME = 0.01   # 走一步花的时间


# 建立Q-table
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions
    )
    # 打印Q-table
    #print(table)
    return table


# 随机选择动作
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]  # 选择该状态下所有可能动作

    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        if state_actions.argmax() == 0:  # 把比较大的值对应的标签赋给action_name
            action_name = 'left'
        else:
            action_name = 'right'
    # print(action_name)

    return action_name


# 获取环境反馈
def get_env_feedback(s, a):  # S的可能情况为：0,1,2,3,4,5
    if a == 'right':
        # s = 4,再向右移动就会到达终点
        if s == N_STATES - 2:
            s1 = 'terminal'
            r = 1
        else:
            s1 = s + 1
            r = 0
    else:
        r = 0

        if s == 0:
            s1 = s
        else:
            r = 0
            s1 = s - 1
    return s1, r


# 搭建环境
def update_env(s, episode, step_counter):
    env_list = ['-']*(N_STATES-1) + ['T']
    if s == 'terminal':
        interaction = 'Episode %s: total_step = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                      ', end='')
    else:
        env_list[s] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)
    # print(env_list)


# 主循环
def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        s = 0
        is_terminated = False
        update_env(s, episode, step_counter)
        while not is_terminated:
            a = choose_action(s, q_table)
            s1, r = get_env_feedback(s, a)
            q_predict = q_table.loc[s, a]
            if s1 != 'terminal':
                q_target = r + LAMBDA * q_table.iloc[s1, :].max()
            else:
                q_target = r
                is_terminated = True

            q_table.loc[s, a] += ALPHA * (q_target - q_predict)
            s = s1

            update_env(s, episode, step_counter+1)
            step_counter += 1
        global EPSILON
        EPSILON += 0.02
        # print(q_table)

    return q_table


if __name__ == "__main__":
    q_table = rl()
    print("\r\nQ-table:\n")
    print(q_table)
