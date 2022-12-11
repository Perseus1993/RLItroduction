import numpy as np
import matplotlib.pyplot as plt


def infi_variance(runs, num_method):
    runs_num = runs
    for _ in range(num_method):
        method_run = runs
        # back = 0, end = 1
        target_policy = 0
        ros = []
        rws = []
        while method_run > 0:
            print(method_run)
            trajectory = []
            while True:
                # behavior policy
                action = np.random.choice(2)
                trajectory.append(action)
                # 直接end了，且得到0的reward  or  0.1概率 end 且得到1的reward
                cur_ro = 0
                if action == 1:
                    # 全是 0
                    ros.append(cur_ro)
                    rws.append(0)
                    method_run -= 1
                    break
                elif np.random.choice([i for i in range(10)]) == 9:
                    # 这个里面全是0(back)
                    ros.append(1.0 / pow(0.5, len(trajectory)))
                    rws.append(1)
                    method_run -= 1
                    break
        ros_arr = np.asarray(ros)
        rws_arr = np.asarray(rws)
        rw_acc = np.add.accumulate(ros_arr * rws_arr)
        ordinary_sampling_trace = rw_acc / np.arange(1, runs_num + 1)
        plt.plot(ordinary_sampling_trace)

    plt.ylim(0,2)
    plt.show()


if __name__ == '__main__':
    infi_variance(100000, 10)
