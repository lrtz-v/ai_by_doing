import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

df = pl.read_csv("challenge-1-beijing.csv")
features = df[
    ["公交", "写字楼", "医院", "商场", "地铁", "学校", "建造时间", "楼层", "面积"]
]
# print(features.describe())

target = df["每平米价格"]  # 目标值数据

split_num = int(len(features) * 0.7)  # 得到 70% 位置

X_train = features[:split_num]  # 训练集特征
y_train = target[:split_num]  # 训练集目标

X_test = features[split_num:]  # 测试集特征
y_test = target[split_num:]  # 测试集目标

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


model = LinearRegression()  # 建立模型
model.fit(X_train, y_train)  # 训练模型
print(model.coef_, model.intercept_)  # 输出训练后的模型参数和截距项

preds = model.predict(X_test)  # 输入测试集特征进行预测
print(preds)  # 预测结果


print(
    "scikit-learn MAE: ", mean_absolute_error(y_test, preds)
)  # 绝对误差的平均值, MAE 的值越小，说明模型拥有更好的拟合程度
print(
    "scikit-learn MSE: ", mean_squared_error(y_test, preds)
)  # 误差的平方的期望值, MSE 的值越小，说明预测模型拥有更好的精确度
print("scikit-learn MAPE: ", mean_absolute_percentage_error(y_test, preds))  # MAPE 值较大,意味着预测的偏移量较大
