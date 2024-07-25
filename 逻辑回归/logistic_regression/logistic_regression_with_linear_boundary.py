# 导入必要的库，用于数据处理和可视化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入自定义的逻辑回归类
from logistic_regression import LogisticRegression

# 读取iris数据集
data = pd.read_csv('../data/iris.csv')
# 查看列名称
# print(data.columns)
# print(data.columns.tolist())
# 定义iris种类的列表
iris_types = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']

# 定义x轴和y轴的特征名称
x_axis = 'petal_length'
y_axis = 'petal_width'

# print(data['class'].head())
# print(set(data['class']))
# 绘制每种iris类型的散点图
for iris_type in iris_types:
    # 获取class列里面数据等于iris_type的petal_length 和  petal_width 数据
    plt.scatter(data[x_axis][data['class'] == iris_type],
                data[y_axis][data['class'] == iris_type],
                label=iris_type
                )
plt.legend()
plt.show()

# 获取数据集的大小
# 多少行（x轴形状）
num_examples = data.shape[0]
# 将特征和标签转换为numpy数组，并进行reshape以适应训练函数的输入格式
# 获取这两列所有数据
# print(data[[x_axis, y_axis]])
# 转成二维数组
# print(data[[x_axis, y_axis]].values)
# reshape 转成指定的维度
x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))
# print(x_train.shape)
y_train = data['class'].values.reshape((num_examples, 1))

# 初始化逻辑回归模型的参数
max_iterations = 1000
polynomial_degree = 0
sinusoid_degree = 0

# 创建逻辑回归实例
logistic_regression = LogisticRegression(x_train, y_train, polynomial_degree, sinusoid_degree)
# 训练模型并获取优化参数和代价历史
thetas, cost_histories = logistic_regression.train(max_iterations)
labels = logistic_regression.unique_labels

# 绘制训练过程中代价函数的变化曲线
plt.plot(range(len(cost_histories[0])), cost_histories[0], label=labels[0])
plt.plot(range(len(cost_histories[1])), cost_histories[1], label=labels[1])
plt.plot(range(len(cost_histories[2])), cost_histories[2], label=labels[2])
plt.legend()
plt.show()

# 对训练数据进行预测，并计算精度
y_train_prections = logistic_regression.predict(x_train)
precision = np.sum(y_train_prections == y_train) / y_train.shape[0] * 100
print('precision:', precision)

# 为了可视化决策边界，定义一个网格
x_min = np.min(x_train[:, 0])
x_max = np.max(x_train[:, 0])
y_min = np.min(x_train[:, 1])
y_max = np.max(x_train[:, 1])
samples = 150
X = np.linspace(x_min, x_max, samples)
Y = np.linspace(y_min, y_max, samples)

# 初始化用于存储预测结果的数组
Z_SETOSA = np.zeros((samples, samples))
Z_VERSICOLOR = np.zeros((samples, samples))
Z_VIRGINICA = np.zeros((samples, samples))

# 在网格上进行预测，并根据预测结果填充数组
for x_index, x in enumerate(X):
    for y_index, y in enumerate(Y):
        data = np.array([[x, y]])
        prediction = logistic_regression.predict(data)[0][0]
        if prediction == 'SETOSA':
            Z_SETOSA[x_index][y_index] = 1
        elif prediction == 'VERSICOLOR':
            Z_VERSICOLOR[x_index][y_index] = 1
        elif prediction == 'VIRGINICA':
            Z_VIRGINICA[x_index][y_index] = 1

# 绘制散点图和决策边界的轮廓
for iris_type in iris_types:
    plt.scatter(
        x_train[(y_train == iris_type).flatten(), 0],
        x_train[(y_train == iris_type).flatten(), 1],
        label=iris_type
    )
plt.contour(X, Y, Z_SETOSA)
plt.contour(X, Y, Z_VERSICOLOR)
plt.contour(X, Y, Z_VIRGINICA)
plt.legend()
plt.show()
