from scipy import stats
import matplotlib.pyplot as plt

def view(lam):
    X = range(0, 20)
    Y = []
    for k in X:
        p = stats.poisson.pmf(k, lam)
        Y.append(p)

    plt.bar(X, Y, color="red")
    plt.xlabel("time")
    plt.ylabel("prob")
    plt.title(f"lam = {lam}")
    plt.show()


if __name__ == '__main__':
    view(2)
    view(3)
    view(4)