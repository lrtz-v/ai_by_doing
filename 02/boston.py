import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pl.read_csv("course-5-boston.csv")
# print(df.head())
# CRIM: 城镇犯罪率。
# ZN: 占地面积超过 2.5 万平方英尺的住宅用地比例。
# INDUS: 城镇非零售业务地区的比例。
# CHAS: 查尔斯河是否经过 (=1 经过，=0 不经过)。
# NOX: 一氧化氮浓度（每 1000 万份）。
# RM: 住宅平均房间数。
# AGE: 所有者年龄。
# DIS: 与就业中心的距离。
# RAD: 公路可达性指数。
# TAX: 物业税率。
# PTRATIO: 城镇师生比例。
# BLACK: 城镇的黑人指数。
# LSTAT: 人口中地位较低人群的百分数。
# MEDV: 城镇住房价格中位数。

features = df[["crim", "rm", "lstat"]]
# print(features.describe())
# ┌────────────┬──────────┬──────────┬───────────┐
# │ statistic  ┆ crim     ┆ rm       ┆ lstat     │
# │ ---        ┆ ---      ┆ ---      ┆ ---       │
# │ str        ┆ f64      ┆ f64      ┆ f64       │
# ╞════════════╪══════════╪══════════╪═══════════╡
# │ count      ┆ 506.0    ┆ 506.0    ┆ 506.0     │
# │ null_count ┆ 0.0      ┆ 0.0      ┆ 0.0       │
# │ mean       ┆ 3.593761 ┆ 6.284634 ┆ 12.653063 │
# │ std        ┆ 8.596783 ┆ 0.702617 ┆ 7.141062  │
# │ min        ┆ 0.00632  ┆ 3.561    ┆ 1.73      │
# │ 25%        ┆ 0.08199  ┆ 5.885    ┆ 6.93      │
# │ 50%        ┆ 0.25915  ┆ 6.209    ┆ 11.38     │
# │ 75%        ┆ 3.67367  ┆ 6.625    ┆ 16.96     │
# │ max        ┆ 88.9762  ┆ 8.78     ┆ 37.97     │
# └────────────┴──────────┴──────────┴───────────┘

target = df["medv"]  # 目标值数据

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


mae_ = mean_absolute_error(y_test, preds)
mse_ = mean_squared_error(y_test, preds)

print("scikit-learn MAE: ", mae_)
print("scikit-learn MSE: ", mse_)
