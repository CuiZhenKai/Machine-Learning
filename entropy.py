'''划分函数：划分数据集'''
def split(X,y,d,value):
    index_a = (X[:,d] <= value)
    index_b = (X[:,d] > value)
    return X[index_a],X[index_a],y[index_a],y[index_b]


'''信息熵计算函数'''
from collections import Counter
from math import log
def entropy(y):
    counter = Counter(y)
    res = 0.0
    for num in counter.values():
        p = num / len(y)
        res += -p * log(p)
    return res


'''寻找最优的划分点'''
def try_split(X,y):
    best_entropy = float('inf')
    best_d = 0.0,best_v = 0.0

    for d in range(X.shape[1]):
        sorted_index = np.argsort(X[:,d])
        for i in range(len(X)):
            if X[sorted_index[i-1],d] != X[sorted_index[i],d]:
                v = (X[sorted_index[i-1],d] + X[sorted_index[i],d]) / 2
                x_l,x_r,y_l,y_r = split(X,y,d,v)
                e = entropy(y_r) + entropy(y_l)
                if e < best_entropy:
                    best_entropy,best_d,best_v = e,d,v


    return best_entropy,best_d,best_v