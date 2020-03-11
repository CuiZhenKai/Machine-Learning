import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def accuracy_score(y_true,y_predict):
    '''check'''
    assert y_true.shape[0] == y_predict.shape[0],\
            "the size must be valid"
    return int((np.sum(y_true == y_predict) / len(y_true)) * 100)

def mean_squared_error(y_true,y_predict):
    '''check'''
    assert len(y_true) == len(y_predict),\
            "the size must be valid"
    return np.sum((y_true - y_predict)**2) / len(y_true)

def root_mean_squared_error(y_true,y_predict):
    return sqrt(mean_squared_error(y_true,y_predict))

def mean_absolute_error(y_true,y_predict):
    '''check'''
    assert len(y_true) == len(y_predict), \
        "the size must be valid"
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)

def r2_score(y_true,y_predict):
    return 1 - mean_absolute_error(y_true,y_predict) / np.var(y_true)

def plot_learning_curve(algo,X_train,y_train,X_test,y_test):
    train_score = []
    test_score = []

    for i in range(1,len(X_train) + 1):
        algo.fit(X_train[:i],y_train[:i])

        y_train_predict = algo.predict(X_train[:i])
        train_score.append(mean_squared_error(y_train[:i],y_train_predict))

        y_test_train = algo.predict(X_test[:i])
        test_score.append(mean_squared_error(y_test[:i],y_test_train))

    plt.plot([i for i in range(1,len(X_train) + 1)],np.sqrt(train_score),label="train")
    plt.plot([i for i in range(1, len(X_train) + 1)], np.sqrt(test_score), label="test")

    plt.legend()
    plt.axis([0,len(X_train)+1,0,4])
    plt.show()

def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1)
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)
