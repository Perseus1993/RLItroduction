import matplotlib.pyplot as plt
import numpy as np

# 左边结束-1, 19个state的真实value
value_truth = np.arange(-19, 0) / 20
gamma = 1


def n_step_td(value_table, step, alpha):
    # 最中间的点是10
    state = 10
    states = []
    rewards = []
    # time step
    ts = 0

    def update_state_reward(st, rw):
        rewards.append(rw)
        states.append(st)
    def update(n_step, cur_time, target_time):
        # 定位reward
        
        tmp = 0
        for i in range(n_step):
            tmp += pow(gamma, i) * rewards[-1 - i]
        state_update = states[-1 * step]
        value_table[state_update] += alpha * (tmp - value_table[state_update])

    def td():
        # 判断是否结束了
        if state in [0, 20]:
            # 剩下的全都算下
            for j in range(step):

        else:
            tmp = 0
            for i in range(3):
                tmp += pow(gamma, i) * rewards[-1 - i]
            state_update = states[-1 * step]
            value_table[state_update] += alpha * (tmp - value_table[state_update])

    while True:
        if state == 0:
            update_state_reward(state, -1)
        action = np.random.choice(2)
        if action == 0:
            state -= 1
        else:
            state += 1
        update_state_reward(state, 0)

        ts += 1
        # update n step之前的state
        if ts >= step:
            td()


def estimate():
    for ep in range(2):
        # 两个终点state加上
        value_table = np.zeros(19 + 2)
        n_step_td(value_table, 3, 0.3)
        print(value_table)


if __name__ == '__main__':
    estimate()
