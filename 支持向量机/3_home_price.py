from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 加载波士顿房价数据集
boston = datasets.load_boston()
# X 代表特征，y 代表目标值（房价）
X = boston.data
y = boston.target

# 将数据集划分为训练集和测试集，测试集占30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建一个支持向量回归（SVR）模型，使用径向基函数（RBF）核
svr_rbf = SVR(kernel='rbf', C=1, gamma=0.1)
# 使用训练集拟合模型
svr_rbf.fit(X_train, y_train)

# 使用测试集进行预测
y_pred_rbf = svr_rbf.predict(X_test)

# 计算模型的性能指标：均方误差（MSE）、平均绝对误差（MAE）和决定系数（R^2）
mse = mean_squared_error(y_test, y_pred_rbf)
mae = mean_absolute_error(y_test, y_pred_rbf)
r2 = r2_score(y_test, y_pred_rbf)
# 打印性能指标的结果
print(f"Mean Squared Error (RBF Kernel): {mse:.2f}")
print(f"Mean Absolute Error (RBF Kernel): {mae:.2f}")
print(f"R-squared Score (RBF Kernel): {r2:.2f}")

# 绘制实际值与预测值的散点图，用于直观评估模型的预测能力
plt.scatter(y_test, y_pred_rbf, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values (SVR - RBF Kernel)")
plt.show()
