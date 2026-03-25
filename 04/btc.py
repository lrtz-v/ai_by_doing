import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def fit_func(p, x):
    """根据公式，定义 n 次多项式函数"""
    f = np.poly1d(p)
    return f(x)


df = pl.read_csv("challenge-2-bitcoin.csv")

data = df.select(
    ["Date", "btc_total_bitcoins", "btc_transaction_fees", "btc_market_price"]
)


def show_data():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(data["Date"], data["btc_market_price"], "r")
    axes[0].set_title("btc_market_price")

    axes[1].plot(data["Date"], data["btc_total_bitcoins"], "r")
    axes[1].set_title("btc_total_bitcoins")

    axes[2].plot(data["Date"], data["btc_transaction_fees"], "r")
    axes[2].set_title("btc_transaction_fees")
    plt.show()


split_num = int(len(data) * 0.7)
train_df = data[:split_num]
test_df = data[split_num:]


# 定义训练和测试使用的自变量和因变量
X_train = np.array(train_df.select(["btc_total_bitcoins", "btc_transaction_fees"]))
y_train = np.array(train_df["btc_market_price"])

X_test = np.array(test_df.select(["btc_total_bitcoins", "btc_transaction_fees"]))
y_test = np.array(test_df["btc_market_price"])

print(len(X_train), len(y_train), len(X_test), len(y_test))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


def poly_plot(N):
    """
    参数:
    N -- 标量, 多项式次数

    返回:
    mse -- N 次多项式预测结果的 MSE 评价指标列表
    """

    m = 1
    mse = []

    ### 代码开始 ### (≈ 6 行代码)
    while m <= N:
        model = make_pipeline(
            PolynomialFeatures(m, include_bias=False), LinearRegression()
        )
        model.fit(X_train, y_train)
        pre_y = model.predict(X_test)
        mse.append(mean_squared_error(y_test, pre_y.flatten()))
        m = m + 1

    ### 代码结束 ###

    return mse


N = 10
mse = poly_plot(10)
print("MSE 计算结果: ", mse)
# 绘图
plt.plot([i for i in range(1, N + 1)], mse, "r")
plt.scatter([i for i in range(1, N + 1)], mse)

# 绘制图名称等
plt.title("MSE of m degree of polynomial regression")
plt.xlabel("m")
plt.ylabel("MSE")
plt.show()
