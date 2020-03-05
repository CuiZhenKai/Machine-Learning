import numpy as np
from math import sqrt

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