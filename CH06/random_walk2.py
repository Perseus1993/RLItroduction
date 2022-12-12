import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

VALUES = np.zeros(7)
VALUES[1:6] = 0.5
VALUES[6] = 1

# set up true state values
TRUE_VALUE = np.zeros(7)
TRUE_VALUE[1:6] = np.arange(1, 6) / 6.0
TRUE_VALUE[6] = 1

ACTION_LEFT = 0
ACTION_RIGHT = 1


def td(values, alpha):
    state = 3
    while True:
        old_state = state
        action = np.random.choice(2)
        if action == 0:
            state -= 1
        else:
            state += 1
        reward = 0
        values[old_state] += alpha * (reward + values[state] - values[old_state])
        if state == 6 or state == 0:
            break


def rms_error():
    # Same alpha value can appear in both arrays
    td_alphas = [0.15, 0.1, 0.05]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    episodes = 100 + 1
    runs = 100
    for alpha in [0.05]:
        total_errors = np.zeros(episodes)
        for r in range(runs):
            print(r)
            errors = []
            current_values = np.copy(VALUES)
            for i in range(0, episodes):
                errors.append(np.sqrt(np.sum(np.power(TRUE_VALUE - current_values, 2)) / 5.0))
                td(current_values, alpha=alpha)
            print(errors)
            print(total_errors)
            total_errors += np.asarray(errors)
        total_errors /= runs
        plt.plot(total_errors, label=', $\\alpha$ = %.02f' % alpha)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    rms_error()
