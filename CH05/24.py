import gym
import numpy as np


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


def mc_estimate(env, runs):
    # 策略是  玩家在手牌点数之和小于 20 时要牌，否则停牌   根据gym的规则 1 要  0不要
    play_policy = [1 for _ in range(23)]
    play_policy[20] = 0
    play_policy[21] = 0

    state_count = np.zeros((10, 10)) + 0.000001
    state_count_without_ace = np.zeros((10, 10)) + 0.000001

    # player 牌是 12到21    庄家亮牌 是 1 - 10
    state_ace = np.zeros((10, 10))
    state_without_ace = np.zeros((10, 10))
    while runs > 0:
        # 记录游戏过程
        observation = env.reset()

        while True:
            action = play_policy[observation[0]]
            old_observation = observation
            observation, reward, done, _, _ = env.step(action)
            if not old_observation[2]:
                state_count_without_ace[old_observation[0] - 12, old_observation[1] - 1] += 1
                state_without_ace[old_observation[0] - 12, old_observation[1] - 1] += reward
            else:
                state_count[old_observation[0] - 12, old_observation[1] - 1] += 1
                state_ace[old_observation[0] - 12, old_observation[1] - 1] += reward

            if done:
                break

        runs -= 1

    return state_ace / state_count, state_without_ace / state_count_without_ace


def mc_exploring(env, runs):
    state_count = np.zeros((10, 10, 2)) + 0.000001
    state_count_without_ace = np.zeros((10, 10, 2)) + 0.000001
    state_ace = np.zeros((10, 10, 2))
    state_without_ace = np.zeros((10, 10, 2))

    # 初始化policy
    policy = np.zeros((10, 10, 2))

    while runs > 0:
        # 记录游戏过程
        episode = []

        # 随机 state action
        observation = env.reset()
        action = np.random.randint(0, 2)

        while True:
            print(runs)

            old_observation = observation
            observation, reward, done, _ = env.step(action)
            episode.append([old_observation, action, reward])
            action = 1 if policy[min(observation[0] - 12, 9), observation[1] - 1, 1] > policy[
                min(observation[0] - 12, 9), observation[1] - 1, 0] else 0
            print("action =", action)
            if done:

                G = 0
                # 完事了，经验倒放
                episode.reverse()
                state_set = []
                for i in episode:

                    cur_state = [i[0][0] - 12, i[0][1] - 1]
                    cur_action = i[1]
                    cur_state.append(cur_action)
                    G += i[2]
                    cur_state.append(i[0][2])
                    if cur_state not in state_set:
                        state_set.append(cur_state)

                        # 有ace
                        if i[0][2]:

                            state_count[cur_state[0], cur_state[1], cur_action] += 1
                            state_ace[cur_state[0], cur_state[1], cur_action] += G
                            policy[cur_state[0], cur_state[1], cur_action] = state_ace[cur_state[0], cur_state[
                                1], cur_action] / state_count[cur_state[0], cur_state[1], cur_action]
                        else:

                            state_count_without_ace[cur_state[0], cur_state[1], cur_action] += 1
                            state_without_ace[cur_state[0], cur_state[1], cur_action] += G
                            policy[cur_state[0], cur_state[1], cur_action] = \
                                state_without_ace[cur_state[0], cur_state[1], cur_action] / state_count[
                                    cur_state[0], cur_state[1], cur_action]

                break
            # policy 产生 action

        runs -= 1

    return policy


if __name__ == '__main__':
    env = gym.make("Blackjack-v1")
    print(mc_estimate(env, 100000))
