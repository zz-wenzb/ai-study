# 导入必要的库，用于数据处理、绘图和线性回归实现
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 从自定义模块导入线性回归类
from linear_regression import LinearRegression

# 读取非线性回归数据集
data = pd.read_csv('../data/non-linear-regression-x-y.csv')

# 将数据集中的x和y值分别reshape为适合处理的格式
x = data['x'].values.reshape((data.shape[0], 1))
y = data['y'].values.reshape((data.shape[0], 1))

# 展示数据集的前10行，用于初步了解数据
data.head(10)

# 绘制原始数据点图，用于直观展示数据分布
plt.plot(x, y)
plt.show()

# 定义训练参数，包括迭代次数、学习率、多项式和正弦函数的次数，以及是否归一化数据
num_iterations = 50000
learning_rate = 0.02
polynomial_degree = 15
sinusoid_degree = 15
normalize_data = True

# 初始化线性回归模型
linear_regression = LinearRegression(x, y, polynomial_degree, sinusoid_degree, normalize_data)

# 训练模型，并获取训练过程中成本函数的历史记录
(theta, cost_history) = linear_regression.train(
    learning_rate,
    num_iterations
)

# 打印训练开始和结束时的成本函数值
print('开始损失: {:.2f}'.format(cost_history[0]))
print('结束损失: {:.2f}'.format(cost_history[-1]))

# 将模型参数 theta 展示为 DataFrame 格式，方便查看
theta_table = pd.DataFrame({'Model Parameters': theta.flatten()})

# 绘制梯度下降过程中成本函数的变化图
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent Progress')
plt.show()

# 预测新的y值，用于展示模型的预测能力
predictions_num = 1000
x_predictions = np.linspace(x.min(), x.max(), predictions_num).reshape(predictions_num, 1)
y_predictions = linear_regression.predict(x_predictions)

# 绘制训练数据点和模型的预测曲线
plt.scatter(x, y, label='Training Dataset')
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')
plt.show()
