import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mping
import math

# 导入自定义的多层感知器类
from multilayer_perceptron import MultilayerPerceptron

# 读取MNIST数据集的示例数据
data = pd.read_csv('../data/mnist-demo.csv')
# 定义要显示的数字数量
numbers_to_display = 25
# 计算图像网格的大小
num_cells = math.ceil(math.sqrt(numbers_to_display))
# 创建一个大的图像网格
plt.figure(figsize=(10, 10))
# 循环遍历每个数字，显示其图像和标签
for plot_index in range(numbers_to_display):
    # 提取当前数字的数据
    digit = data[plot_index:plot_index + 1].values
    # 获取数字的标签
    digit_label = digit[0][0]
    # 获取数字的像素值
    digit_pixels = digit[0][1:]
    # 计算图像的大小
    image_size = int(math.sqrt(digit_pixels.shape[0]))
    # 将像素值重塑为图像的形状
    frame = digit_pixels.reshape((image_size, image_size))
    # 在网格中创建一个子图，并显示图像
    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(frame, cmap='Greys')
    plt.title(digit_label)
# 调整子图之间的间距，并显示图像
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

# 从数据集中划分训练集和测试集
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

# 将数据集转换为numpy数组格式
train_data = train_data.values
test_data = test_data.values

# 定义训练集的大小
num_training_examples = 5000

# 分割特征和标签
x_train = train_data[:num_training_examples, 1:]
y_train = train_data[:num_training_examples, [0]]

x_test = test_data[:, 1:]
y_test = test_data[:, [0]]

# 定义神经网络的层数和每层的神经元数量
layers = [784, 25, 10]

# 是否对数据进行归一化
normalize_data = True
# 训练的最大迭代次数
max_iterations = 500
# 学习率
alpha = 0.1

# 创建多层感知器实例
multilayer_perceptron = MultilayerPerceptron(x_train, y_train, layers, normalize_data)
# 训练神经网络
(thetas, costs) = multilayer_perceptron.train(max_iterations, alpha)
# 绘制训练过程中成本函数的变化
plt.plot(range(len(costs)), costs)
plt.xlabel('Grident steps')
plt.xlabel('costs')
plt.show()

# 对训练集和测试集进行预测
y_train_predictions = multilayer_perceptron.predict(x_train)
y_test_predictions = multilayer_perceptron.predict(x_test)

# 计算训练集和测试集的准确率
train_p = np.sum(y_train_predictions == y_train) / y_train.shape[0] * 100
test_p = np.sum(y_test_predictions == y_test) / y_test.shape[0] * 100
print('训练集准确率：', train_p)
print('测试集准确率：', test_p)

# 定义要显示的数字数量
numbers_to_display = 64

# 计算图像网格的大小
num_cells = math.ceil(math.sqrt(numbers_to_display))

# 创建一个大的图像网格
plt.figure(figsize=(15, 15))

# 循环遍历每个数字，显示其图像和预测的标签
for plot_index in range(numbers_to_display):
    # 获取数字的标签
    digit_label = y_test[plot_index, 0]
    # 获取数字的像素值
    digit_pixels = x_test[plot_index, :]
    # 获取模型预测的标签
    predicted_label = y_test_predictions[plot_index][0]

    # 计算图像的大小
    image_size = int(math.sqrt(digit_pixels.shape[0]))

    # 将像素值重塑为图像的形状
    frame = digit_pixels.reshape((image_size, image_size))

    # 根据预测的标签是否正确，选择颜色映射
    color_map = 'Greens' if predicted_label == digit_label else 'Reds'

    # 在网格中创建一个子图，并显示图像
    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(frame, cmap=color_map)
    plt.title(predicted_label)
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# 调整子图之间的间距，并显示图像
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()
