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

    def fit_gd(self,X_train,y_train,eta=0.01,n_iters=1e4):
        '''check'''
        assert X_train.shape[0] == y_train.shape[0],\
                "the size must be valid"

        '''No.1 get J'''
        def J(X_b,theta,y):
            try:
                return np.sum(y - X_b.dot(theta)) / len(y)
            except:
                return float('inf')


        def dJ_debug(theta,X_b,y,epsilon):
            res = np.empty(len(theta))
            for i in range(res):
                theta_1 = theta.copy()
                theta_1[i] += epsilon
                theta_2 = theta.copy()
                theta_2[i] -= epsilon
                res[i] = (J(X_b,theta_1,y) - J(X_b,theta_2,y))  / (2*epsilon)

            return res

        def DJ(X_b,theta,y):
            return X_b.T.dot(X_b.dot(theta) - y) *2 / len(X_b)

        def gradient_decent(X_b,y,initial_theta,eta,n_iters = 1e4,epsilon = 1e-8):
            theta = initial_theta
            cur_iters = 0

            while cur_iters < n_iters:
                gradient = DJ(X_b,y,theta)
                last_theta = theta
                theta = theta - gradient*n_iters
                if(abs(J(X_b,y,theta) - J(X_b,y,last_theta)) < epsilon):
                    break

                cur_iters += 1

        X_b = np.hstack([np.ones((len(X_train),1)),X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta  = gradient_decent(X_b,y_train,initial_theta,eta,n_iters)
        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]
        return self

    def fit_sgd(self,X_train,y_train,eta=0.01,n_iters=5,t0=5,t1=50):
        '''check'''
        assert X_train.shape[0] == y_train.shape[0],\
                "the size must be valid"
        assert n_iters>=1,\
                "the valud must be >= 1"

        def dJ_sgd(theta,x_b_i,y_i):
            return x_b_i*(x_b_i.dot(theta) - y_i) * 2.

        def sgd(X_b,y,initial_theta,n_iters,t0=5,t1=50):
            def learning_rate(t):
                return t0 / (t + t1)
            theta = initial_theta
            m = len(X_b)

            for cur_iters in range(n_iters):
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes]
                y_train_new = y_train[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta,X_b_new[i],y_train_new[i])
                    theta = theta - learning_rate(cur_iters * m +i) * gradient

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, eta, n_iters)
        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]
        return self

    def fit_mbsgd(self,X_train,y_train,eta=0.01,n_iters = 5,t0=5,t1=50):
        '''check'''
        assert X_train.shape[0] == y_train.shape[0],\
                "the size must be valid"
        assert n_iters>=1,\
                "the value must be >=1"

        def dJ_sgd(theta,X_b_i,y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2

        def mbsgd(X_b,y,initial_theta,n_iters,n=100,t0=5,t1=50):
            def learning_rate(t):
                return t0 / (t + t1)

            theta = initial_theta

            for cur_iter in range(n_iters):
                for i in range(n):
                    rand_i = []
                    rand_i_new = np.random.randint(len(X_b))
                    rand_i.append(rand_i_new)

                gradient = dJ_sgd(theta,X_b[rand_i],y_train[rand_i])
                theta = theta - learning_rate(cur_iter) * gradient
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = mbsgd(X_b, y_train, initial_theta, eta, n_iters)
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


class LogisticRegression:

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    '''tool function'''
    def _sigmod(self,t):
        return 1. / (1. + np.exp(-t))

    '''use the gradient to fit the model'''
    def fit(self,X_train,y_train,eta=0.01,n_iters = 1e4):

        '''check'''
        assert X_train.shape[0] == y_train.shape[0],\
                "the size must be valid"

        def J(theta,X_b,y):
            y_hat = self._sigmod(X_b.dot(theta))
            try:
                return -np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) / len(y)
            except:
                return float('inf')

        def DJ(theta,X_b,y):
            return X_b.T.dot(self._sigmod(X_b.dot(theta)) - y) / len(X_b)

        def gradient_decent(X_b,y,initial_theta,eta,n_iters=1e4,epsilon = 1e-8):
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                last_theta = theta
                gradient = DJ(theta,X_b,y)
                theta = theta - eta*gradient
                while(abs(J(theta,X_b,y) - J(last_theta,X_b,y))):
                    break
                cur_iter += 1

            return theta

        '''main step'''
        X_b = np.hstack([np.ones((len(X_train),1)),X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_decent(X_b,y_train,initial_theta,eta,n_iters)
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]
        return self

    def predict_proda(self,X_predict):
        '''check'''
        assert self.coef_ is not None and self.intercept_ is not None,\
                "predict after fit"
        assert X_predict.shape[1] == len(self.coef_),\
                "the size must be valid"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self._sigmod(X_b.dot(self._theta))

    '''real predict'''
    def predict(self,X_predict):
        '''check'''
        assert self.coef_ is not None and self.intercept_ is not None, \
            "predict after fit"
        assert X_predict.shape[1] == len(self.coef_), \
            "the size must be valid"

        proda = self.predict_proda(X_predict)
        return np.array(proda>=0.5,dtype='int')

    def __repr__(self):
        return "LogisticRegression()"




