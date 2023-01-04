import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    env = Env()
    # value function
    values = np.zeros(10)
    distribution = np.zeros(1002)
    mc_gradient(env, values, distribution)
    plt.scatter(values)
    true_values = [values[i//100] for i in range(1000)]
    plt.plot(true_values)
    plt.show()
    plt.close()
    plt.plot(distribution/np.sum(distribution))
    plt.show()

