import gym
import numpy as np
import matplotlib.pyplot as plt


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
    run_num = runs
    # ground truth
    truth = -0.27726

    # target policy
    play_policy = [1 for _ in range(23)]
    play_policy[20] = 0
    play_policy[21] = 0

    init_state = (13, 2, True)

    # ro & reward
    ros = []
    rws = []

    while runs > 0:
        print(runs)

        trajectory = []
        observation = init_state
        while True:
            # 查了 open ai的 文档不能指定开始state，只能这样搞了
            while env.reset()[0] != init_state:
                pass

            # behavior policy = 0.5要牌，0.5不跟
            action = np.random.choice(2)
            trajectory.append((observation, action))
            new_observation, reward, done, _, _ = env.step(action)
            if done:
                runs -= 1
                ro_cur = 1
                for oba in trajectory:
                    # 计算ro， 照着那个公式，分子是每个target policy 在 state的出现 a的概率，分子是behavior policy 在 state的出现 a的概率
                    ro_cur *= (1 if play_policy[oba[0][0]] == oba[1] else 0) / 0.5
                ros.append(ro_cur)
                rws.append(reward)
                break
            observation = new_observation
    ros_arr = np.asarray(ros)
    rws_arr = np.asarray(rws)
    rw_acc = np.add.accumulate(ros_arr * rws_arr)

    # 防止 0 除 0 报错
    np.seterr(divide='ignore', invalid='ignore')
    ordinary_sampling_trace = rw_acc / np.arange(1, run_num + 1)

    # ros累计
    ros_acc = np.add.accumulate(ros_arr)
    weighted_sampling_trace = rw_acc / ros_acc

    mse = np.power(ordinary_sampling_trace - truth, 2)
    mse2 = np.power(weighted_sampling_trace - truth, 2)

    plt.plot(mse)
    plt.plot(mse2)
    plt.show()


if __name__ == '__main__':
    env = gym.make("Blackjack-v1")
    mc_off_policy(env, 1000)
