import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Env:
    def step(self, state, action):
        # action -1 左 1 右
        next_pos = state + action * (np.random.choice(100) + 1)
        # 判断出界
        next_pos = 1001 if next_pos > 1001 else next_pos
        next_pos = 0 if next_pos < 0 else next_pos
        if next_pos == 0:
            return True, -1, next_pos
        if next_pos == 1001:
            return True, 1, next_pos

        return False, 0, next_pos


def mc_gradient(env, values, distribution):
    actions = [1, -1]
    for ep in range(100000):
        trajectory = []
        state = 500
        while True:
            action = np.random.choice(actions)
            is_end, reward, next_state = env.step(state, action)
            trajectory.append(next_state)
            state = next_state
            if is_end:
                break
        # 更新除了end state 之外的
        for state in trajectory[:-1]:
            group_num = (state - 1) // 100
            # print(state, group_num)
            values[group_num] += 0.00002 * (reward - values[group_num])
            distribution[state] += 1


# n step td
def semi_gradient_td(env, approximate, n):
    actions = [1, -1]
    for ep in tqdm(range(100000)):
        state = 500
        states = [state]
        rws = [0]
        t = 0
        T = np.inf
        while True:
            t += 1
            if t < T:
                action = np.random.choice(actions)
                is_end, reward, next_state = env.step(state, action)
                states.append(next_state)
                rws.append(reward)
                if is_end:
                    T = t
            tao = t - n
            if tao >= 0:
                G = 0
                for nt in range(tao + 1, min(tao + n, T) + 1):
                    G += rws[nt]
                if tao + n < T:
                    G += approximate.get_value(states[tao + n])
                update_state = states[tao]
                if update_state not in (0, 1001):
                    uv = 0.00002 * (G - approximate.get_value(update_state))
                    approximate.update(uv, update_state)
            if tao == T - 1:
                break
            state = next_state


def semi_gradient_td_train():
    approximate = Approximate(10, 1000)
    semi_gradient_td(env, approximate, 2)
    print(approximate.param)


class Approximate:
    def __init__(self, group_num, state_num):
        self.group_num = group_num
        self.state_num = state_num
        self.param = np.zeros(group_num)

    def get_value(self, state):
        # print(int((state - 1) // (self.state_num / self.group_num)))
        return self.param[int((state - 1) // (self.state_num / self.group_num))]

    def update(self, value, state):
        self.param[int((state - 1) // (self.state_num / self.group_num))] += value


if __name__ == '__main__':
    env = Env()
    # # value function
    # values = np.zeros(10)
    # distribution = np.zeros(1002)
    # mc_gradient(env, values, distribution)
    # plt.scatter(values)
    # true_values = [values[i // 100] for i in range(1000)]
    # plt.plot(true_values)
    # plt.show()
    # plt.close()
    # plt.plot(distribution / np.sum(distribution))
    # plt.show()
    semi_gradient_td_train()
