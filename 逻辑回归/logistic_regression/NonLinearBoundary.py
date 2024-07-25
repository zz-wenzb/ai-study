# 导入必要的库，用于数据处理和可视化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# 导入自定义的逻辑回归类
from logistic_regression import LogisticRegression

# 读取微芯片测试数据
data = pd.read_csv('../data/microchips-tests.csv')

# 定义有效标签值
# 类别标签
validities = [0, 1]

# 选择用于可视化两个特征的参数
# 选择两个特征
x_axis = 'param_1'
y_axis = 'param_2'

# 绘制数据点，根据有效性分为两组
# 散点图
for validity in validities:
    plt.scatter(
        data[x_axis][data['validity'] == validity],
        data[y_axis][data['validity'] == validity],
        label=validity
    )
  
# 图例和标签
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.title('Microchips Tests')
plt.legend()
plt.show()

# 获取数据集大小
num_examples = data.shape[0]

# 准备训练数据，reshape以适应逻辑回归模型
x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))
y_train = data['validity'].values.reshape((num_examples, 1))

# 初始化逻辑回归模型参数
# 训练参数
max_iterations = 100000  
regularization_param = 0  
polynomial_degree = 5  
sinusoid_degree = 0  

# 实例化逻辑回归模型
# 逻辑回归
# 存在的问题 polynomial_degree有值的话，后面会有个变量作用域报错
logistic_regression = LogisticRegression(x_train, y_train, polynomial_degree, sinusoid_degree)

# 训练模型
# 训练
(thetas, costs) = logistic_regression.train(max_iterations)

# 为成本函数绘制图形
columns = []
for theta_index in range(0, thetas.shape[1]):
    columns.append('Theta ' + str(theta_index));

# 训练结果
labels = logistic_regression.unique_labels

plt.plot(range(len(costs[0])), costs[0], label=labels[0])
plt.plot(range(len(costs[1])), costs[1], label=labels[1])

# 图例和标签
plt.xlabel('Gradient Steps')
plt.ylabel('Cost')
plt.legend()
plt.show()

# 使用训练好的模型进行预测
# 预测
y_train_predictions = logistic_regression.predict(x_train)

# 计算预测的精度
# 准确率
precision = np.sum(y_train_predictions == y_train) / y_train.shape[0] * 100

print('Training Precision: {:5.4f}%'.format(precision))

# 为了可视化决策边界，准备一个网格
num_examples = x_train.shape[0]
samples = 150
x_min = np.min(x_train[:, 0])
x_max = np.max(x_train[:, 0])

y_min = np.min(x_train[:, 1])
y_max = np.max(x_train[:, 1])

X = np.linspace(x_min, x_max, samples)
Y = np.linspace(y_min, y_max, samples)
Z = np.zeros((samples, samples))

# 在网格上进行预测，以创建决策边界
# 结果展示
for x_index, x in enumerate(X):
    for y_index, y in enumerate(Y):
        data = np.array([[x, y]])
        Z[x_index][y_index] = logistic_regression.predict(data)[0][0]

# 绘制数据点和决策边界
positives = (y_train == 1).flatten()
negatives = (y_train == 0).flatten()

plt.scatter(x_train[negatives, 0], x_train[negatives, 1], label='0')
plt.scatter(x_train[positives, 0], x_train[positives, 1], label='1')

plt.contour(X, Y, Z)

# 图例和标签
plt.xlabel('param_1')
plt.ylabel('param_2')
plt.title('Microchips Tests')
plt.legend()

plt.show()
