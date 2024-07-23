from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 生成一个线性回归问题的数据集
X, y = make_regression(n_samples=100, n_features=1, n_informative=1, noise=10, random_state=0)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 创建一个SGDRegressor实例并设置参数
sgd_reg = SGDRegressor(loss="squared_loss", learning_rate="constant", eta0=0.1)
# 训练模型
sgd_reg.fit(X_train, y_train)

# 预测测试集
y_pred = sgd_reg.predict(X_test)
print(y_test)
print(y_pred)
# 评估模型
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
