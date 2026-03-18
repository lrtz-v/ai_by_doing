import numpy as np
from common import f, square_loss, x, y


def least_squares_matrix(x: np.matrix, y: np.matrix) -> np.matrix:
    """最小二乘法矩阵求解"""
    w = (x.T * x).I * x.T * y
    return w


x_matrix = np.matrix(np.hstack((np.ones((x.shape[0], 1)), x.reshape(x.shape[0], 1))))
y_matrix = np.matrix(y.reshape(y.shape[0], 1))
w = least_squares_matrix(x_matrix, y_matrix)
print(w)
w0, w1 = w[0][0], w[1][0]
loss = square_loss(x, y, w0, w1)
print(f"w0:{w0}, w1:{w1}, loss:{loss}")
print(f"price when area is 150: {f(150, w0, w1)}")
