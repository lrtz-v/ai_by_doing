import numpy as np
from sklearn.linear_model import Ridge


def ridge_regression(X, Y, alpha):
    """
    参数:
    X -- 自变量数据矩阵
    Y -- 因变量数据矩阵
    alpha -- lamda 参数

    返回:
    W -- 岭回归系数
    """
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X, Y)
    return ridge.coef_


np.random.seed(10)  # 设置随机数种子
X = np.matrix(np.random.randint(5, size=(10, 10)))
Y = np.matrix(np.random.randint(10, size=(10, 1)))
alpha = 0.5


print(ridge_regression(np.asarray(X), np.asarray(Y), alpha).T)
