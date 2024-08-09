import numpy as np

# 数据生成
np.random.seed(0)
X = 2 * np.random.rand(100, 2) - 1  # 100个样本，每个样本2个特征
y = (np.dot(X, np.array([0.5, 0.3])) + -0.2 + 0.1 * np.random.randn(100)) > 0  # 真实逻辑回归模型为y = 0.5x1 + 0.3x2 - 0.2 + 噪声

# 参数初始化
theta = np.zeros(X.shape[1])  # 初始参数设置为0


# 定义Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 定义损失函数及其梯度
def loss_function(theta, X, y):
    predictions = sigmoid(np.dot(X, theta))
    errors = predictions - y
    gradient = np.dot(X.T, errors) / len(X)
    return np.sum(errors ** 2) / len(errors), gradient


# 梯度上升优化
learning_rate = 0.1
iterations = 1000
for i in range(iterations):
    loss, gradient = loss_function(theta, X, y)
    theta -= learning_rate * gradient  # 参数更新
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss}, Theta: {theta}")

# 模型预测
predictions = sigmoid(np.dot(X, theta)) > 0.5
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy}")
