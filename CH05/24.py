import gym
import numpy as np
import matplotlib.pyplot as plt


def play(env):
    observation = env.reset()
    print("观测={}".format(observation))
    while True:
        print("玩家={}，庄家={}，".format(env.player, env.dealer))
        action = np.random.choice(env.action_space.n)
        print("动作={}".format(action))
        observation, reward, done = env.step(action)
        print("观测={}，奖励={}，结束指示={}".format(observation, reward, done))
        if done:
            break


def mc_on_policy(env, runs):
    # 策略是  玩家在手牌点数之和小于 20 时要牌，否则停牌   根据gym的规则 1 要  0不要
    play_policy = [1 for _ in range(23)]
    play_policy[20] = 0
    play_policy[21] = 0

    state_count = np.zeros((10, 10)) + 0.000001
    state_count_without_ace = np.zeros((10, 10)) + 0.000001

    # player 牌是 12到21    庄家亮牌 是 1 - 10
    state_ace = np.zeros((10, 10))
    state_without_ace = np.zeros((10, 10))

    # 初始化returns,100个state
    returns = [[] * 10] * 10
    while runs > 0:
        # 记录游戏过程
        observation = env.reset()[0]
        trajectory = []

        while True:
            action = play_policy[observation[0]]
            new_observation, reward, done, _, _ = env.step(action)
            trajectory.append(observation)

            if done:
                runs -= 1
                for ob in trajectory:
                    print(ob)
                    if ob[2]:
                        state_count[ob[0] - 12, ob[1] - 1] += 1
                        state_ace[ob[0] - 12, ob[1] - 1] += reward
                    else:
                        state_count_without_ace[ob[0] - 12, ob[1] - 1] += 1
                        state_without_ace[ob[0] - 12, ob[1] - 1] += reward

                break
            observation = new_observation

    return state_ace / state_count, state_without_ace / state_count_without_ace


def mc_off_policy(env, runs):
    init_state = (13, 2)
    while runs > 0:
        ro = []
        trajectory = []
        while True:
            action = np.random.choice(2)





if __name__ == '__main__':
    env = gym.make("Blackjack-v1")

