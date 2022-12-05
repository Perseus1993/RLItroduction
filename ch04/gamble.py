import numpy as np

h_prob = 0.4


def gamble():
    values = np.zeros(101)
    values[100] = 1.0

    while True:
        old_values = values.copy()

        # 1块钱到99块钱
        for i in range(1, 100):

            # 下注范围
            actions = np.arange(min(100 - i, i) + 1)
            action_returns = []
            for action in actions:
                money = h_prob * values[i + action] + (1 - h_prob) * values[i - action]
                action_returns.append(money)
            values[i] = np.max(action_returns)

        if abs(values - old_values).max() < 1e-4:
            break

    policy = np.zeros(101)
    for i in range(1, 100):
        actions = np.arange(min(100 - i, i) + 1)
        action_returns = []
        for action in actions:
            action_returns.append(h_prob * values[i + action] + (1 - h_prob) * values[i - action])

        policy[i] = np.argmax(action_returns) + 1

    print(policy)


if __name__ == '__main__':
    gamble()
