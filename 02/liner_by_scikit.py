from common import x, y
from sklearn.linear_model import LinearRegression

# 定义线性回归模型
model = LinearRegression()
model.fit(
    x.reshape(x.shape[0], 1), y
)  # 训练, reshape 操作把数据处理成 fit 能接受的形状

# 得到模型拟合参数
print(model.intercept_, model.coef_)
print(f"price when area is 150: {model.predict([[150]])}")
