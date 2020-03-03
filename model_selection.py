import numpy as np

def train_test_split(X,y,test_train = 0.2,seed = None):
    '''check'''
    assert X.shape[0] == y.shape[0],\
            "the size must be valid"
    assert 0.0 <= test_train <= 1.0,\
            "the ratio must be in 0-1"

    if seed:
        np.random.seed(seed)

    shuffle_index= np.random.permutation(len(X))

    test_ratio = test_train
    test_size = int(len(X) * test_ratio)

    test_indexes = X[:test_size]
    train_indxes = X[test_size:]

    X_train = X[train_indxes]
    X_test = X[test_indexes]

    y_train = y[train_indxes]
    y_test = y[test_indexes]

    return X_train,X_test,y_train,y_test
