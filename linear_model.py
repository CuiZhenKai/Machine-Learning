import numpy as np
'''value the model precision'''
from sklearn.metrics import r2_score

class SimpleLinearRegression:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self,X_train,y_train):
        '''check'''
        assert X_train.ndim == 1,"the dimension must be 1"
        assert len(X_train)== len(y_train),\
                "the size must be valid"

        x_mean = np.mean(X_train)
        y_mean = np.mean(y_train)

        self.a_ = ((X_train - x_mean).dot(y_train - y_mean)) / ((X_train - x_mean).dot(X_train - x_mean))
        self.b_ = y_mean - (self.a_ * x_mean)

        return self

    def predict(self,X_test):
        '''check'''
        assert self.a_ is not None and self.b_ is not None,\
                "predict after fit"
        assert X_test.ndim == 1,"the dimension must be 1"

        return np.array([self._predict(x) for x in X_test])

    def _predict(self,x):
        return self.a_ * x + self.b_

    def score(self,X_test,y_test):
        y_predict = self(X_test)
        return r2_score(y_test,y_predict)

    def __repr__(self):
        return "SimpleLinearRegression()"


class LinearRegression:
    def __init__(self):
        self._theta = None
        self.interception_ = None
        self.coef_ = None

    def fit_normal(self,X_train,y_train):
        '''check'''
        assert X_train.shape[0] == y_train.shape[0],\
                "the size must be valid"

        X_b = np.hstack([np.ones((len(X_train),1)),X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]
        return self

    def predict(self,X_predict):
        '''check'''
        assert self.interception_ is not None and self.coef_ is not None,\
                "fit before predict"
        assert X_predict.shape[1] = len(self.coef_),\
                "the size must be valid"

        X_b = np.hstack([np.ones((len(X_predict),1)),X_predict])
        return X_b.dot(self._theta)

    def score(self,X_test,y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test,y_predict)

    def __repr__(self):
        return "LinearRegression()"


