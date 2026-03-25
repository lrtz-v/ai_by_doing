import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression



x = [4, 8, 12, 25, 32, 43, 58, 63, 69, 79]
y = [20, 33, 50, 56, 42, 31, 33, 46, 65, 75]
x = np.array(x).reshape(len(x), 1)  # 转换为列向量
y = np.array(y).reshape(len(y), 1)

# 使用 sklearn 得到 2 次多项式回归特征矩阵
poly_features = PolynomialFeatures(degree=2, include_bias=False)
# - degree: 多项式次数，默认为 2 次多项式
# - include_bias: 默认为 True，包含多项式中的截距项。
poly_x = poly_features.fit_transform(x)


# 定义线性回归模型
model = LinearRegression()
model.fit(poly_x, y)  # 训练

# 绘制拟合图像
x_temp = np.array(x).reshape(len(y), 1)
poly_x_temp = poly_features.fit_transform(x_temp)

plt.plot(x_temp, model.predict(poly_x_temp), "r")
plt.scatter(x, y)
plt.show()
