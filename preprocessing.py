import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self,X):
        '''according to X get mean and std'''
        '''check'''
        assert X.ndim == 2,"the dimension must be 2"

        self.mean_ = np.array([np.mean[:,i] for i in range(0,X.shape[1])])
        self.std_ = np.array([np.std[:,i] for i in range(0,X.shape[1])])

        return self
    def transform(self,X):
        '''check'''
        assert self.mean_ is not None and self.std_ is not None,\
                "fit before transform"
        assert X.ndim == 2,"the dimension must be 2"
        assert X.shape[1] == len(self.mean_),\
                "the value must be valid"

        resX = np.empty(shape=X.shape,dtype=float)
        for col in range(X.shape[1]):
            resX[:,col] = (X[:,col] - self.mean_[col]) / self.std_[col]

        return resX