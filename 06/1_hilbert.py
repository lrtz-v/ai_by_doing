import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import hilbert
from scipy.optimize import leastsq
from sklearn.linear_model import Lasso, Ridge

"""
y= w_{1} * x_{1} + w_{2} * x_{2} +……+w_{10} * x_{10}\tag{7}
"""
x = hilbert(10)  # 生成 10x10 的希尔伯特矩阵
np.random.seed(10)  # 随机数种子能保证每次生成的随机数一致

w = np.random.randint(2, 10, 10)  # 随机生成 w 系数
y_temp = np.matrix(x) * np.matrix(w).T  # 计算 y 值
y = np.array(y_temp.T)[0]  # 将 y 值转换成 1 维行向量


print("实际参数 w: ", w)
print("实际函数值 y: ", y)


def func(p, x):
    return np.dot(x, p)  # 函数公式


def err_func(p, x, y):
    return func(p, x) - y  # 残差函数


p_init = np.random.randint(1, 2, 10)  # 全部参数初始化为 1

parameters = leastsq(err_func, p_init, args=(x, y))  # 最小二乘法求解
print("拟合参数 w: ", parameters[0])


"""
使用岭回归，引入正则化(L2正则项)，解决最小二乘法回归的局限
"""

"""不同 alpha 参数拟合"""


def rudge():
    alphas = np.linspace(1, 10, 20)

    coefs = []
    for a in alphas:
        ridge = Ridge(alpha=a, fit_intercept=False)
        ridge.fit(x, y)
        coefs.append(ridge.coef_)
    print(coefs)

    plt.plot(alphas, coefs)  # 绘制不同 alpha 参数下的 w 拟合值
    plt.scatter(
        np.linspace(1, 0, 10), parameters[0]
    )  # 普通最小二乘法拟合的 w 值放入图中
    plt.xlabel("alpha")
    plt.ylabel("w")
    plt.title("Ridge Regression")
    plt.show()
    # 当 alpha 取值越大时，正则项主导收敛过程，各系数趋近于 0。当 alpha 很小时，各系数波动幅度变大。


"""
如果特征变量间的相关性较强，则可能会导致某些系数很大，而另一些系数变成很小的负数。
使用LASSO回归，引入正则化(L1 正则项)，解决最小二乘法回归的局限
"""


def lasso():
    alphas = np.linspace(1, 10, 10)
    lasso_coefs = []

    for a in alphas:
        lasso = Lasso(alpha=a, fit_intercept=False)
        lasso.fit(x, y)
        lasso_coefs.append(lasso.coef_)

    plt.plot(alphas, lasso_coefs)  # 绘制不同 alpha 参数下的 w 拟合值
    plt.scatter(
        np.linspace(1, 0, 10), parameters[0]
    )  # 普通最小二乘法拟合的 w 值放入图中
    plt.xlabel("alpha")
    plt.ylabel("w")
    plt.title("Lasso Regression")
    plt.show()
    # 当 alpha 取值越大时，正则项主导收敛过程，各系数趋近于 0。当 alpha 很小时，各系数波动幅度变大。

lasso()
