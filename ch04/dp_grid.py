import numpy as np


class Map:
    def __init__(self, height, width):
        self.height = height
        self.width = width

        # 元组之类的不好相加，使用np转array,判断用list，运算用array
        self.a_point = [0, 1]
        self.b_point = [0, 3]

    def step(self, state, action):
        # 定义终点
        if state == [0, 0] or state == [3, 3]:
            return state, 0

        next_state = (np.array(state) + np.array(action))
        # 判断越界
        if next_state[0] < 0 or next_state[1] < 0 or next_state[0] >= self.height or next_state[1] >= self.width:
            return state, -1

        # 正常行走
        return next_state.tolist(), -1


actions = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def train(map):
    value = np.zeros((map.height, map.width))
    iter_num = 0
    while True:
        old_value = value.copy()
        for i in range(map.height):
            for j in range(map.width):
                this_value = 0
                for action in actions:
                    new_state, reward = map.step([i, j], action)
                    this_value += 0.25 * (reward + value[new_state[0], new_state[1]])
                value[i, j] = this_value

        if abs(old_value - value).max() < 0.0001:
            print(iter_num, value)
            break
        iter_num += 1


if __name__ == '__main__':
    m = Map(4, 4)
    train(m)
