import matplotlib.pyplot as plt
import numpy as np

# 左边结束-1, 19个state的真实value
value_truth = np.arange(-19, 0) / 20


def n_step_td(value_table, step, alpha):
    state = 9
    states = []
    reward = 0
    while True:
        if state == -1:
            reward = -1
            break
        elif state == 19:
            break
        action = np.random.choice(2)
        if action == 0:
            state -= 1
        else:
            state += 1
        states.append(state)


def estimate():
    # 两个终点state加上
    value_table = np.zeros(19 + 2)


if __name__ == '__main__':
    n_step_td()
