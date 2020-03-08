import numpy as np

class PCA:

    def __init__(self,n_components):
        '''check'''
        assert n_components >= 1,\
                "the value must be >=1"
        self.n_components  = n_components
        self.components = None

    def fit(self,X,eta=0.01,n_iters=1e4):
        '''check'''
        assert self.n_components <= X.shape[1],\
                "the size must be valid"

        '''No.1 demean'''
        def demean(x):
            return x - np.mean(X,axis = 0)

        '''No.2 '''
        def f(w,x):
            return np.sum((X.dot(w)) **2 ) / len(x)

        '''No.3 Df'''
        def df(w,x):
            return x.T.dot(x.dot(w)) * 2 / len(x)

        '''No.4 direction'''
        def direction(w):
            return w / np.linalg.norm(w)

        '''No.5 get first component'''
        def first_component(X,initial_w,eta=0.01,n_iters = 1e4,epsilon = 1e-8):
            w = initial_w
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = df(w,X)
                last_w = w
                w = w + eta*gradient
                if(abs(f(w,X) - f(last_w,X)) < epsilon):
                    break
                cur_iter+=1

            return w

        '''main step'''
        X_demean = demean(X)
        self.components = np.empty(shape=(self.n_components,X.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(X_demean.shape[1])
            w = first_component(X,initial_w)
            self.components[i,:] = w

            X_demean = X_demean - X_demean.dot(w).reshape(-1,1) * w

        return self

    def transform(self,X):
        '''check'''
        assert X.shape[1] == self.components.shape[1],\
                "the size must be valid"

        return X.dot(self.components.T)

    def inverse_transform(self, X):
        '''check'''
        assert X.shape[1] == self.components.shape[0], \
                "the size must be valid"

        return X.dot(self.components)

    def __repr__(self):
        '''desc'''
        return "PCA(n_components=%d)" % self.n_components
