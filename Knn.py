import numpy as np
from collections import Counter
from math import sqrt

class KNeighborsClassifier:

    def __init__(self,k):
        '''check'''
        assert k>=1,"K must be valid"
        self.k = k

        '''private'''
        self._X_train = None
        self._y_train = None

    def fit(self,X,y):
        '''check'''
        assert X.shape[0] == y.shape[0],\
                "The data size must be valid"
        assert self.k <= y.shape[0],\
                "K must be valid"

        self._X_train = X
        self._y_train = y

        '''according to the scikit-learn,we need to return self'''
        return self

    def predict(self,x_predict):
        '''check'''
        assert self._X_train.shape[1] == x_predict.shape[1],\
                "The size must be valid"
        assert self._X_train is not None and self._y_train is not None,\
                "The process fit should do before predict"

        y_predict = [self._doPredict(x) for x in x_predict]
        return np.array(y_predict)

    def _doPredict(self,x):
        '''check'''
        assert x.shape[0] == self._X_train.shape[1],\
                "The size must be valid"

        distances = [sqrt(np.sum((x - x_train)**2)) for x_train in self._X_train]
        sortedIndex = np.argsort(distances)
        topK = [self._y_train[i] for i in sortedIndex[:self.k]]
        return Counter(topK).most_common(1)[0][0]

    def __repr__(self):
        return "KNeighborsClassifier()"



