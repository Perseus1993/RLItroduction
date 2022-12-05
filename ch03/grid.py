import numpy as np


class Map:
    def __init__(self, height, width):
        self.height = height
        self.width = width

        # 元组之类的不好相加，使用np转array,判断用list，运算用array
        self.a_point = [0, 1]
        self.b_point = [0, 3]

    def step(self, state, action):
        # A,B
        if state == self.a_point:
            new_state = (4, 1)
            reward = 10
            return new_state, reward
        if state == self.b_point:
            new_state = (2, 3)
            reward = 5
            return new_state, reward

        next_state = (np.array(state) + np.array(action))
        # 判断越界
        if next_state[0] < 0 or next_state[1] < 0 or next_state[0] >= self.height or next_state[1] >= self.width:
            return state, -1

        # 正常行走
        return next_state.tolist(), 0


actions = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def random(map):
    value = np.zeros((map.height, map.width))
    while True:
        new_value = np.zeros_like(value)
        for i in range(map.height):
            for j in range(map.width):
                for action in actions:
                    state, reward = map.step([i, j], action)
                    new_value[i, j] += 0.25 * (reward + 0.9 * value[state[0], state[1]])
                    print(i, "-", j, "reward ", reward)

        if np.sum(np.abs(value - new_value)) < 1e-4:
            print(value)
            break
        value = new_value


def random_linear(map):
    # 构建reward矩阵 25*1
    reward_m = np.zeros((map.height * map.width, 1))

    # 构建prob矩阵 25*25
    prob_m = np.zeros((map.height * map.width, map.height * map.width))
    for i in range(map.height):
        for j in range(map.width):
            for action in actions:
                new_state, reward = map.step([i, j], action)
                # 获取老state在25*25向量的相对位置
                state_num = np.ravel_multi_index(np.array([i, j]), (map.height, map.width))
                # 获取新state在25*25向量的相对位置
                new_state_num = np.ravel_multi_index(np.array(new_state), (map.height, map.width))

                # 如果是边界,prob在回弹到当前格子会累加
                prob_m[state_num, new_state_num] += 0.25
                reward_m[np.ravel_multi_index(np.array([i, j]), (map.height, map.width))] += reward

    return np.dot(np.linalg.pinv(1 - 0.9 * prob_m), reward_m)


def optimal(map):
    q_values = np.zeros((map.height, map.width))
    iter_num = 0

    while True:
        new_q_values = np.zeros_like(q_values)
        for i in range(map.height):
            for j in range(map.width):
                action_values = []
                for action in actions:
                    new_state, reward = map.step([i, j], action)
                    action_values.append(reward + 0.9 * q_values[new_state[0], new_state[1]])
                new_q_values[i, j] = np.max(action_values)

        diff = np.sum(np.abs(new_q_values - q_values))

        if diff < 0.001:
            print(q_values)
            break

        print("iter= ", iter_num, " diff= ", diff)
        iter_num += 1

        q_values = new_q_values


if __name__ == '__main__':
    m = Map(5, 5)
    optimal(m)
