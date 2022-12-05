import numpy as np
from scipy.stats import poisson


# 停车场，可以接受车辆
class ParkingLot:
    def __init__(self):
        self.car_num = 0

    # 车辆数超过20抹去
    def get_car(self, num):
        self.car_num += num
        if self.car_num > 20:
            self.car_num = 20
        return self.car_num


# 只与数字相关
def income(park_num, prob):
    # 租车收入 = sum((车保有量 - k)* poisson.pmf(k,lambda)) * 10
    income_from_rent = 0

    for i in range(park_num + 1):
        cur_prob = prob[i]
        income_from_rent += cur_prob * (park_num - i) * 10

    return income_from_rent


# 租车还车的联合分布表
def init_prob_tables(lam1, lam2, len1, len2):
    prob_table = np.zeros((len1, len2))
    for i in range(len1):
        for j in range(len2):
            prob_table[i, j] = poisson.pmf(i, lam1) * poisson.pmf(j, lam2)
    return prob_table


# 最多调配5辆车
actions = [i for i in range(-5, 6)]


def validate_car_num(nums_in):
    return min(nums_in, 20)


# 传入的action事先判定了，不会引起负数
def compute_profit(park1_num, park2_num, action, values, prob_table_request, prob_table_return):
    # 要返回的v(state,action)估计
    cur_value = 0

    # 挪车
    cur_value = -2 * abs(action)
    reward = 0
    car_num1 = validate_car_num(park1_num - 1 * action)
    car_num2 = validate_car_num(park2_num + action)

    # 租出去
    for i in range(min(int(car_num1), 13)):
        for j in range(min(int(car_num2), 13)):
            prob1 = prob_table_request[i, j]
            reward += prob1 * 10 * (i + j)
            car_num1_after_rent = validate_car_num(car_num1 - i)
            car_num2_after_rent = validate_car_num(car_num2 - j)

            # 还车,影响的是下一天的state,也就是s'

            for ii in range(13):
                for jj in range(13):
                    prob2 = prob_table_return[ii, jj] * prob1
                    next_state1 = validate_car_num(car_num1_after_rent + ii)
                    next_state2 = validate_car_num(car_num2_after_rent + jj)

                    # print(i, j, prob2, reward, next_state1, next_state2)
                    cur_value += prob2 * (reward + 0.9 * values[next_state1, next_state2])
    return cur_value


if __name__ == '__main__':
    park1 = ParkingLot()
    park2 = ParkingLot()

    # policy初始化，值是从park1到park2的调车数量，负的就是反向,这里不规定整数后面会变成分数，奇怪
    policy = np.zeros((21, 21), dtype=np.int)
    values = np.zeros((21, 21))

    # 概率表
    prob_request = init_prob_tables(3, 4, 13, 13)
    prob_return = init_prob_tables(2, 2, 13, 13)

    while True:
        while True:
            old_values = values.copy()
            for i in range(21):
                for j in range(21):
                    values[i, j] = compute_profit(i, j, policy[i, j], values, prob_request, prob_return)

            delta = abs(old_values - values).max()
            print('max delta {}'.format(delta))
            if delta < 1e-4:
                break

        policy_remain = True
        for i in range(21):
            for j in range(21):
                old_action = policy[i, j]
                action_values = []
                for action in actions:
                    if action > i or action < -1 * j:
                        action_values.append(-1 * np.inf)
                    else:
                        action_values.append(compute_profit(i, j, action, values, prob_request, prob_return))
                improved_action = actions[np.argmax(action_values)]
                policy[i, j] = improved_action
                if improved_action != old_action:
                    print("policy in state", i, j, "change from ", old_action, " to ", improved_action)
                    policy_remain = False

        if policy_remain:
            print(policy)
            break
