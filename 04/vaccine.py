import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

df = pl.read_csv("course-6-vaccine.csv")

# 首先划分 dateframe 为训练集和测试集
split_num = int(len(df) * 0.7)
train_df = df[:split_num]
test_df = df[split_num:]

# 定义训练和测试使用的自变量和因变量
X_train = np.array(train_df["Year"])
y_train = np.array(train_df["Values"])

X_test = np.array(test_df["Year"])
y_test = np.array(test_df["Values"])


# # 建立线性回归模型
# model = LinearRegression()
# model.fit(X_train.reshape(len(X_train), 1), y_train.reshape(len(y_train), 1))
# results = model.predict(X_test.reshape(len(X_test), 1))
# print(results)  # 线性回归模型在测试集上的预测结果

# print("线性回归平均绝对误差: ", mean_absolute_error(y_test, results.flatten()))
# print("线性回归均方误差: ", mean_squared_error(y_test, results.flatten()))


# # 2 次多项式回归特征矩阵
# poly_features_2 = PolynomialFeatures(degree=2, include_bias=False)
# poly_X_train_2 = poly_features_2.fit_transform(X_train.reshape(len(X_train), 1))
# poly_X_test_2 = poly_features_2.fit_transform(X_test.reshape(len(X_test), 1))

# # 2 次多项式回归模型训练与预测
# model = LinearRegression()
# model.fit(poly_X_train_2, y_train.reshape(len(X_train), 1))  # 训练模型

# results_2 = model.predict(poly_X_test_2)  # 预测结果

# print("2 次多项式回归平均绝对误差: ", mean_absolute_error(y_test, results_2.flatten()))
# print("2 次多项式均方误差: ", mean_squared_error(y_test, results_2.flatten()))


###############################
# 更高次多项式回归预测 && 项式回归预测次数选择

X_train = X_train.reshape(len(X_train), 1)
X_test = X_test.reshape(len(X_test), 1)
y_train = y_train.reshape(len(y_train), 1)

# 计算 m 次多项式回归预测结果的 MSE 评价指标并绘图
mse = []  # 用于存储各最高次多项式 MSE 值
m = 1  # 初始 m 值
m_max = 10  # 设定最高次数
while m <= m_max:
    model = make_pipeline(PolynomialFeatures(m, include_bias=False), LinearRegression())
    model.fit(X_train, y_train)
    pre_y = model.predict(X_test)
    print(
        "{} 次多项式回归平均绝对误差: ".format(m),
        mean_absolute_error(y_test, pre_y.flatten()),
    )
    print(
        "{} 次多项式均方误差: ".format(m), mean_squared_error(y_test, pre_y.flatten())
    )
    print("---")
    mse.append(mean_squared_error(y_test, pre_y.flatten()))  # 计算 MSE
    m += 1

print("MSE 计算结果: ", mse)
# 绘图
plt.plot([i for i in range(1, m_max + 1)], mse, "r")
plt.scatter([i for i in range(1, m_max + 1)], mse)

# 绘制图名称等
plt.title("MSE of m degree of polynomial regression")
plt.xlabel("m")
plt.ylabel("MSE")
plt.show()
