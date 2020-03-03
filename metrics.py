import numpy as np

def accuracy_score(y_true,y_predict):
    '''check'''
    assert y_true.shape[0] == y_predict.shape[0],\
            "the size must be valid"
    return int((np.sum(y_true == y_predict) / len(y_true)) * 100)