import numpy as np
from matplotlib import pyplot as plt

# U D L R
actions = [0, 1, 2, 3]
move = [(-1, 0), (1, 0), (0, -1), (0, 1)]
epsilon = 0.1
alpha = 0.2


class Env:
    height = 4
    width = 12
    terminal = (3, 11)
    start = (3, 0)

    def step(self, state, action):

        next_state = np.array(state) + np.array(move[action])
        # 判断出界
        if next_state[0] < 0:
            next_state[0] = 0
        elif next_state[0] >= self.height:
            next_state[0] = self.height - 1
        elif next_state[1] < 0:
            next_state[1] = 0
        elif next_state[1] >= self.width:
            next_state[1] = self.width - 1

        # 判断cliff
        if next_state[0] == 3 and next_state[1] not in [0, 11]:
            return False, -100, (3, 0)
        if next_state[0] == 3 and next_state[1] == 11:
            return True, 0, None

        return False, -1, next_state


def find_action_from_q_table(q_table, state):
    values = q_table[state[0]][state[1]]
    return np.random.choice(np.where(values == np.max(values))[0])


def sarsa(env: Env, q_table, alpha, epsilon):
    state = env.start
    # 首先找出动作
    if np.random.binomial(1, epsilon) == 1:
        action = np.random.choice(4)
    else:
        action = find_action_from_q_table(q_table, state)
    # 记录当前episode的reward
    rewards = 0
    while True:
        is_finish, reward, next_state = env.step(state, action)
        # print("action = ", action, "next state = ", next_state)
        if is_finish:
            break
        rewards += reward
        if np.random.binomial(1, epsilon) == 1:
            next_action = np.random.choice(4)
        else:
            next_action = find_action_from_q_table(q_table, next_state)
        # sarsa update
        q_table[state[0]][state[1]][action] += alpha * (-1 + q_table[next_state[0]][next_state[1]][next_action]
                                                        - q_table[state[0]][state[1]][action])
        action = next_action
        state = next_state
    return rewards


def q_learning(env: Env, q_table, alpha, epsilon):
    state = env.start
    rewards = 0
    while True:
        if np.random.binomial(1, epsilon) == 1:
            action = np.random.choice(4)
        else:
            action = find_action_from_q_table(q_table, state)

        is_finish, reward, next_state = env.step(state, action)
        # print("action = ", action, "next state = ", next_state)
        if is_finish:
            break
        rewards += reward
        next_action = find_action_from_q_table(q_table, next_state)
        # sarsa update
        q_table[state[0]][state[1]][action] += alpha * (-1 + q_table[next_state[0]][next_state[1]][next_action]
                                                        - q_table[state[0]][state[1]][action])
        state = next_state
    return rewards


def train(env):
    episodes = 500
    runs = 50
    rewards_sarsa = np.zeros(episodes)
    rewards_q = np.zeros(episodes)
    for r in range(runs):
        print(r)
        q_table_sa = np.zeros((env.height, env.width, 4))
        q_table_q = np.zeros((env.height, env.width, 4))
        for ep in range(episodes):
            print("--", ep)
            rewards_sarsa[ep] += sarsa(env, q_table_sa, 0.2, 0.1)
            rewards_q[ep] += q_learning(env, q_table_q, 0.2, 0.1)
        if r == runs - 1:
            print(q_table_q)
            print_optimal_policy(q_table_sa, env)
            print_optimal_policy(q_table_q, env)
    rewards_sarsa /= runs
    rewards_q /= runs

    plt.plot(rewards_sarsa, label='sarsa')
    plt.plot(rewards_q, label='q_learning')
    plt.xlabel('episode')
    plt.ylabel('sum reward')
    plt.ylim([-100, 0])
    plt.legend()
    plt.show()


def print_optimal_policy(q_value, env):
    optimal_policy = []
    for i in range(0, env.height):
        optimal_policy.append([])
        for j in range(0, env.width):
            if [i, j] == [3, 11]:
                optimal_policy[-1].append('G')
                continue
            act = np.argmax(q_value[i, j, :])
            if act == 0:
                optimal_policy[-1].append('U')
            elif act == 1:
                optimal_policy[-1].append('D')
            elif act == 2:
                optimal_policy[-1].append('L')
            elif act == 3:
                optimal_policy[-1].append('R')
    for row in optimal_policy:
        print(row)



if __name__ == '__main__':
    e = Env()
    train(e)
