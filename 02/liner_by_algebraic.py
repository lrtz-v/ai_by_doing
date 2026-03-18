import numpy as np
from common import f, square_loss, x, y
from matplotlib import pyplot as plt


def least_squares_algebraic(x: np.ndarray, y: np.ndarray):
    """最小二乘法代数求解"""
    n = x.shape[0]
    w1 = (n * sum(x * y) - sum(x) * sum(y)) / (n * sum(x * x) - sum(x) * sum(x))
    w0 = (sum(x * x) * sum(y) - sum(x) * sum(x * y)) / (
        n * sum(x * x) - sum(x) * sum(x)
    )
    return w0, w1


w0, w1 = least_squares_algebraic(x, y)
loss = square_loss(x, y, w0, w1)
print(f"w0:{w0}, w1:{w1}, loss:{loss}")
print(f"price when area is 150: {f(150, w0, w1)}")
plt.scatter(x, y)
plt.xlabel("Area")
plt.ylabel("Price")
x_temp = np.linspace(50, 120, 100)  # 绘制直线生成的临时点
plt.scatter(x, y)
plt.plot(x_temp, x_temp * w1 + w0, "r")
plt.show()
