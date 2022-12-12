import math

import numpy as np
import matplotlib.pyplot as plt


def td0(init_values, alpha, truth, err):
    value_table = init_values
    cur_state = 3
    while True:
        old_state = cur_state
        action = np.random.choice(2)
        # left 0, right 1
        if action == 0:
            cur_state -= 1
        else:
            cur_state += 1
        reward = 1 if cur_state == 6 else 0
        value_table[old_state] += alpha * (reward + value_table[cur_state] - value_table[old_state])
        # finish ?
        if cur_state in [0, 6]:
            break
            # 计算 rms
    res = 0
    for index in range(1, len(value_table) - 1):
        res += ((truth[index] - value_table[index]) ** 2)
    err.append(math.sqrt(res / 5))
    print("---------------------")
    print(truth[1:6])
    print(value_table[1:6])
    print("err = ", err[-1])


def td_evaluate():
    value_truth = np.array([0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1])

    # value初始化都是 0.5
    values = np.zeros(7)
    values[1:6] = 0.5

    # 评估1,10,100次时候输出
    output = [0, 1, 10, 100]
    for episode in range(101):
        if episode in output:
            plt.plot(values[1:6], label=episode)
        td0(values, 0.1, None, None)
    plt.plot(value_truth[1:6], label='true values')
    plt.legend()
    plt.show()


def mc(init_values, alpha, truth, err):
    value_table = init_values
    cur_state = 3
    trajectory = [cur_state]
    while True:

        action = np.random.choice(2)

        if action == 0:
            cur_state -= 1
        else:
            cur_state += 1
        trajectory.append(cur_state)
        if cur_state in [0, 6]:
            reward = 1 if cur_state == 6 else 0
            break
    # 更新value
    for st in trajectory:
        value_table[st] += alpha * (reward - value_table[st])
    # 计算 rms
    res = 0
    for index in range(len(value_table)):
        res += ((truth[index] - value_table[index]) ** 2)
    err.append(math.sqrt(res / 5))
    print("*************")
    print(truth)
    print(value_table)


def mc_vs_td():
    tda = [0.05, 0.15, 0.1]
    mca = [0.01, 0.02, 0.03, 0.04]
    value_truth = np.array([0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1])

    # value初始化都是 0.5
    values = np.zeros(7)
    values[1:6] = 0.5
    values[6] = 1

    # for a in mca:
    #     # rms_err列表
    #     err = []
    #     cur_value_init = values.copy()
    #     for i in range(101):
    #         mc(cur_value_init, a, value_truth, err)
    #     plt.plot(err, label="mc - " + str(a))
    for a in [0.1]:
        err2 = []
        cur_value_init = values.copy()
        for i in range(101):
            td0(cur_value_init, a, value_truth, err2)
        plt.plot(err2, label="td - " + str(a))

    plt.legend()
    plt.show()


if __name__ == '__main__':
    mc_vs_td()
