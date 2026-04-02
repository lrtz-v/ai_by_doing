import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
from sklearn.linear_model import LinearRegression


def fit_func(p, x):
    """根据公式，定义 n 次多项式函数"""
    f = np.poly1d(p)
    return f(x)


def err_func(p, x, y):
    """残差函数（观测值与拟合值之间的差距）"""
    ret = fit_func(p, x) - y
    return ret


def leastsq_re():
    data = pl.read_csv("advertising.csv")
    tv_df = data["tv"]
    radio_df = data["radio"]
    newspaper_df = data["newspaper"]
    sales_df = data["sales"]

    p_init = np.random.randn(2)
    params_tv = leastsq(err_func, p_init, args=(np.array(tv_df), np.array(sales_df)))
    params_radio = leastsq(
        err_func, p_init, args=(np.array(radio_df), np.array(sales_df))
    )
    params_newspaper = leastsq(
        err_func, p_init, args=(np.array(newspaper_df), np.array(sales_df))
    )

    print(params_tv[0], params_radio[0], params_newspaper[0])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    x_tv = np.array([data["tv"].min(), data["tv"].max()])
    axes[0].plot(x_tv, params_tv[0][0] * x_tv + params_tv[0][1], "r")
    axes[0].scatter(tv_df, sales_df)
    axes[0].set_title("tv")

    x_radio = np.array([data["radio"].min(), data["radio"].max()])
    axes[1].plot(x_radio, params_radio[0][0] * x_radio + params_radio[0][1], "r")
    axes[1].scatter(radio_df, sales_df)
    axes[1].set_title("radio")

    x_newspaper = np.array([data["newspaper"].min(), data["newspaper"].max()])
    axes[2].plot(
        x_newspaper, params_newspaper[0][0] * x_newspaper + params_newspaper[0][1], "r"
    )
    axes[2].scatter(newspaper_df, sales_df)
    axes[2].set_title("newspaper")

    plt.show()


def linearRe():
    df = pl.read_csv("advertising.csv")
    features = df["tv", "radio", "newspaper"]
    target = df["sales"]
    model = LinearRegression()  # 建立模型
    model.fit(features, target)  # 训练模型
    print(model.coef_, model.intercept_)  # 输出训练后的模型参数和截距项


def smf_ols():
    df = pl.read_csv("advertising.csv")
    results = smf.ols(formula="sales ~ tv + radio + newspaper", data=df).fit()
    print(results.summary2())


linearRe()
smf_ols()
