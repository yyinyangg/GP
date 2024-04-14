import numpy as np
from gp import MyGP
import matplotlib.pyplot as plt
SEED = 0


def seed_everything(seed: int):
    np.random.seed(seed)


def run():
    GP = MyGP()
    X_train = np.array([[1], [2], [3]])
    samples = GP.sample(X_train, 30)
    for s in samples:
        plt.plot(range(len(X_train)), s)

    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.title("GP sampling")
    plt.show()



if __name__ == '__main__':
    seed_everything(SEED)
    run()
