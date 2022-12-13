import numpy as np
from matplotlib import pyplot as plt

epsilon = 0.1
alpha = 0.5
actions = np.array([(0, 1), (0, -1), (1, 0), (-1, 0)])


# x 竖着, y 横着
class Env:
    wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    height = 7
    width = 10
    termial = (3, 7)

    def get_position(self, next_state):
        # 判断越界
        if next_state[0] < 0:
            next_state[0] = 0
        elif next_state[1] < 0:
            next_state[1] = 0
        elif next_state[0] >= self.height:
            next_state[0] = self.height - 1
        elif next_state[1] >= self.width:
            next_state[1] = self.width - 1
        return next_state

    def step(self, state, action):
        next_state_no_wind = (np.array(state) + np.array(action))
        next_state_no_wind = self.get_position(next_state_no_wind)
        next_state = (np.array(next_state_no_wind)
                      + np.array((self.wind_strength[next_state_no_wind[1]], 0)))
        next_state = self.get_position(next_state)

        if next_state[0] == self.termial[0] and next_state[1] == self.termial[0]:
            return True, 0, next_state
        else:
            return False, -1, next_state


def episode(q_table, env):
    print("episode --------")
    time = 0
    state = (3, 0)
    if np.random.binomial(1, epsilon) == 1:
        # 随机
        action = actions[np.random.choice(4)]
    else:
        # 查阅q表
        values = q_table[state[0], state[1]]
        max_value = np.max(values)
        action = actions[np.random.choice(np.where(values == max_value)[0])]

    while True:
        print("action is", action)
        is_finish, reward, next_state = env.step(state, action)
        if is_finish:
            print(q_table)
            break
        # next action
        if np.random.binomial(1, epsilon) == 1:
            next_action = actions[np.random.choice(4)]
        else:
            values = q_table[next_state[0]][next_state[1]]
            print("@", np.where(values == np.max(values))[0])
            next_action = actions[np.random.choice(np.where(values == np.max(values))[0])]

        # sarsa
        action_num = np.random.choice(np.where(actions == action)[0])
        next_action_num = np.random.choice(np.where(actions == next_action)[0])
        print(state, ":", q_table[state[0]][state[1]][action_num], "=>", next_state, ":",
              q_table[state[0]][state[1]][action_num])
        print(action_num, next_action_num, reward)
        print("from ", q_table[state[0]][state[1]][action_num])
        print("to ", alpha * (-1 + q_table[next_state[0]][next_state[1]][next_action_num]))
        q_table[state[0]][state[1]][action_num] += alpha * \
                                                   (-1 + q_table[next_state[0]][next_state[1]][next_action_num])
        # print(state[0], state[1], action, actions, np.where(actions == action))
        # print("qt", q_table[state[0]][state[1]][action_num])
        state = next_state
        action = next_action
        time += 1

    return time


def train(env):
    q_table = np.zeros((env.height, env.width, 4))
    episode_limit = 500
    steps = []
    ep = 0
    while ep < episode_limit:
        print(ep)
        steps.append(episode(q_table, env))
        # time = episode(q_value)
        # episodes.extend([ep] * time)
        ep += 1

    steps = np.add.accumulate(steps)

    plt.plot(steps, np.arange(1, len(steps) + 1))
    plt.show()
    print(q_table)

    # display the optimal policy
    optimal_policy = []
    for i in range(0, 7):
        optimal_policy.append([])
        for j in range(0, 10):
            if [i, j] == [3, 7]:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(q_table[i, j, :])
            if bestAction == 2:
                optimal_policy[-1].append('U')
            elif bestAction == 3:
                optimal_policy[-1].append('D')
            elif bestAction == 1:
                optimal_policy[-1].append('L')
            elif bestAction == 0:
                optimal_policy[-1].append('R')
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)
    print('Wind strength for each column:\n{}'.format([str(w) for w in WIND]))


if __name__ == '__main__':
    e = Env()
    train(e)
