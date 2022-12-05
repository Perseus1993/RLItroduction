import numpy as np
import matplotlib.pyplot as plt


# 单个摇杆
class Arm:
    def __init__(self, reward_avg):
        self.reward_avg = reward_avg

    def give_reward(self):
        # 按照reward预设生成数据
        return np.random.normal(self.reward_avg, 1)


# 生成N个摇杆
def generate_arm_list(arm_num):
    reward_avg = np.random.randn(arm_num) + 1
    return [Arm(re) for re in reward_avg]


# 赌博机
class Bandit:
    def __init__(self, arm_list, strategy, epsilon=0.0, ucb_param=0, gradient_step=0.0, optim=False):
        # 摇杆数
        self.k = len(arm_list)
        # 估计表 可用于不同策略
        self.q_estimate = np.zeros(len(arm_list))
        # 策略
        self.strategy = strategy
        self.epsilon = epsilon
        # 摇杆次数统计
        self.arm_count = np.zeros(len(arm_list))

        # 运行次数
        self.time = 0
        self.total_reward = 0
        self.arms = arm_list
        # ucb系数
        self.ucb_param = ucb_param

        # gradient系数
        self.gradient_step = gradient_step
        exp = np.exp(self.q_estimate)
        self.action_prob = exp / np.sum(exp)
        print("初始prob", self.action_prob)

        # 乐观选项
        if optim:
            self.q_estimate = [i + 5 for i in self.q_estimate]

        # 存下最优解,极端情况可能有多个
        best_arm = [0]
        for i, a in enumerate(self.arms):
            print("arm avg = ", a.reward_avg)
            if i != 0 and a.reward_avg == self.arms[best_arm[0]].reward_avg:
                best_arm.append(i)
            elif a.reward_avg > self.arms[best_arm[0]].reward_avg:
                best_arm = [i]

        print("best arm  = ", best_arm)

        self.init()

    # 赌博机根据行为给出reward
    def give_reward(self, action):
        return self.arms[action].give_reward()

    # 重置
    def init(self):
        self.q_estimate = np.zeros(self.k)
        self.arm_count = np.zeros(self.k)

    # 升级q_estimate
    def single_step(self, action):
        self.arm_count[action] += 1
        current_reward = self.give_reward(action)
        self.total_reward += current_reward

        if self.strategy == 'greedy':
            self.q_estimate[action] += (current_reward - self.q_estimate[action]) / self.arm_count[action]
        elif self.strategy == 'ucb':
            # 分母加了0.1 ^ 4 省的初始值是0
            self.q_estimate[action] += self.ucb_param * np.sqrt(
                np.log(self.time) / (self.arm_count[action] + 0.1 ** 4))

        if self.strategy == 'gradient':
            # 这点注意 梯度更新要同步，先要算概率
            exp = np.exp(self.q_estimate)
            self.action_prob = exp / np.sum(exp)

            print(self.action_prob)

            for i in range(len(self.q_estimate)):
                if i == action:
                    self.q_estimate[action] += self.gradient_step * \
                                               (current_reward - (current_reward + self.total_reward) / self.time) * (
                                                       1 - self.action_prob[action])
                else:
                    self.q_estimate[action] -= self.gradient_step * \
                                               (current_reward - (current_reward + self.total_reward) / self.time) * \
                                               self.action_prob[action]

    def choose_action(self):

        if self.strategy == 'greedy':
            # greedy随机选择
            if np.random.rand() < self.epsilon:
                return np.random.choice(self.k)
            else:
                # greedy贪婪选择
                return np.random.choice(np.where(self.q_estimate == np.max(self.q_estimate))[0])
        # ucb直接选最大的
        if self.strategy == 'ucb':
            return np.random.choice(np.where(self.q_estimate == np.max(self.q_estimate))[0])

        if self.strategy == 'gradient':
            return np.random.choice(np.arange(self.k), p=self.action_prob)


def train(bandit, num_iter, print_every_iter):
    reward_record = np.zeros(num_iter)
    for i in range(num_iter):
        bandit.time += 1
        action = bandit.choose_action()
        bandit.single_step(action)
        if i % print_every_iter == 0:
            print(action, "---", bandit.q_estimate, "--- average_reward =", bandit.total_reward / bandit.time)
        reward_record[i] = bandit.total_reward / bandit.time
    return reward_record


if __name__ == '__main__':
    arm_list = generate_arm_list(10)
    b = Bandit(arm_list, strategy='greedy', epsilon=0.01)
    b2 = Bandit(arm_list, strategy='greedy', epsilon=0.1)
    b3 = Bandit(arm_list, strategy='greedy', epsilon=0)
    b4 = Bandit(arm_list, strategy='ucb', ucb_param=2)
    b5 = Bandit(arm_list, strategy='gradient', gradient_step=0.1)
    b6 = Bandit(arm_list, strategy='greedy', epsilon=0.1, optim=True)

    record = train(b, 1000, 10000)
    record2 = train(b2, 1000, 10000)
    record3 = train(b3, 1000, 10000)
    record4 = train(b4, 1000, 10000)
    record5 = train(b5, 1000, 100)
    record6 = train(b6, 1000, 100)

    plt.plot(record, label='ep = 0.01')
    plt.plot(record2, label='ep = 0.1')
    plt.plot(record3, label='ep = 0')
    plt.plot(record4, label='ucb(param = 2)')
    plt.plot(record5, label='gradient(param = 0.1)')
    plt.plot(record6, label='ep = 0.1 and optim')
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()
    plt.show()
