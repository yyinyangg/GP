from gp import MyGP
import matplotlib.pyplot as plt
import numpy as np

def true_function(x):
    return np.sin(x[0])*np.cos(x[1])

def plot(Data, Y, title):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(Data[:,0], Data[:,1], Y, linewidth=0.2, antialiased=True)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title(title)
    plt.show()

if __name__ == "__main__":
    X_train = np.random.uniform(low=-2, high=2, size=(50,2))
    Y_train = true_function(X_train.T)
    gp = MyGP()
    gp.fit(X_train, Y_train)

    X_test = np.array(np.meshgrid(np.linspace(-2,2,10),np.linspace(-2,2,10))).T.reshape(-1,2)
    Y_pre = gp.predict(X_test)
    Y_gt = true_function(X_test.T)

    plot(X_test, Y_pre, 'prediction')
    plot(X_test, Y_gt, 'ground truth')
    plot(X_test, Y_pre - Y_gt, 'errors')
