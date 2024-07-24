# 导入必要的库，用于数据处理和可视化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入自定义的线性回归类
from linear_regression import LinearRegression

# 读取世界幸福报告的数据集
data = pd.read_csv('../data/world-happiness-report-2017.csv')

# 将数据集分为训练集和测试集
# 得到训练和测试数据
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

# 定义输入和输出参数的列名
input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

# 提取训练集和测试集的输入和输出参数
x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values

# 绘制训练集和测试集的数据点
plt.scatter(x_train, y_train, label='Train data')
plt.scatter(x_test, y_test, label='test data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()

# 定义梯度下降的迭代次数和学习率
num_iterations = 500
learning_rate = 0.01

# 初始化线性回归模型
linear_regression = LinearRegression(x_train, y_train)
# 训练模型并获取成本历史记录
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)

# 打印训练前后的成本
print('开始时的损失：', cost_history[0])
print('训练后的损失：', cost_history[-1])

# 绘制成本随迭代次数变化的曲线
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iter')
plt.ylabel('cost')
plt.title('GD')
plt.show()

# 预测新的数据点
predictions_num = 100
x_predictions = np.linspace(x_train.min(), x_train.max(), predictions_num).reshape(predictions_num, 1)
y_predictions = linear_regression.predict(x_predictions)

# 绘制训练集、测试集和预测结果的数据点
plt.scatter(x_train, y_train, label='Train data')
plt.scatter(x_test, y_test, label='test data')
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()
