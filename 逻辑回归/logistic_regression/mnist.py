# 导入必要的库，用于数据处理、图像展示和逻辑回归算法实现
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

# 从自定义模块导入逻辑回归类
from logistic_regression import LogisticRegression

# 读取MNIST数据集的示例数据
data = pd.read_csv('../data/mnist-demo.csv')

# 设置显示的数字数量和子图大小
# 绘图设置
numbers_to_display = 25
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(10, 10))

# 循环显示数字图像及其标签
# 
for plot_index in range(numbers_to_display):
    # 提取当前数字的数据
    # 读取数据
    digit = data[plot_index:plot_index + 1].values
    digit_label = digit[0][0]
    digit_pixels = digit[0][1:]

    # 计算图像的大小
    # 正方形的
    image_size = int(math.sqrt(digit_pixels.shape[0]))
    
    # 将像素数据重塑为图像格式
    # 转换成图像形式
    frame = digit_pixels.reshape((image_size, image_size))
    
    # 在子图中展示图像，并添加标签
    # 展示图像
    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(frame, cmap='Greys')
    plt.title(digit_label)
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# 调整子图间距，并展示所有图像
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()

# 划分训练集和测试集
# 训练集划分
pd_train_data = data.sample(frac=0.8)
pd_test_data = data.drop(pd_train_data.index)

# 将DataFrame数据转换为numpy数组格式
# Ndarray数组格式
train_data = pd_train_data.values
test_data = pd_test_data.values

# 提取训练集和测试集的特征和标签
num_training_examples = 6000
x_train = train_data[:num_training_examples, 1:]
y_train = train_data[:num_training_examples, [0]]

x_test = test_data[:, 1:]
y_test = test_data[:, [0]]

# 设置逻辑回归的参数
# 训练参数
max_iterations = 10000  
polynomial_degree = 0  
sinusoid_degree = 0  
normalize_data = True  

# 初始化逻辑回归模型
# 逻辑回归
logistic_regression = LogisticRegression(x_train, y_train, polynomial_degree, sinusoid_degree, normalize_data)

# 训练模型并获取训练过程中的成本函数值
(thetas, costs) = logistic_regression.train(max_iterations)

# 展示训练得到的参数
pd.DataFrame(thetas)

# 设置展示的数字数量和子图大小
# How many numbers to display.
numbers_to_display = 9

# Calculate the number of cells that will hold all the numbers.
num_cells = math.ceil(math.sqrt(numbers_to_display))

# Make the plot a little bit bigger than default one.
plt.figure(figsize=(10, 10))

# 循环展示逻辑回归参数对应的数字图像
# Go through the thetas and print them.
for plot_index in range(numbers_to_display):
    # 提取当前数字的参数
    # Extrace digit data.
    digit_pixels = thetas[plot_index][1:]

    # 计算图像大小
    # Calculate image size (remember that each picture has square proportions).
    image_size = int(math.sqrt(digit_pixels.shape[0]))
    
    # 将参数重塑为图像格式
    # Convert image vector into the matrix of pixels.
    frame = digit_pixels.reshape((image_size, image_size))
    
    # 在子图中展示图像
    # Plot the number matrix.
    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(frame, cmap='Greys')
    plt.title(plot_index)
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# 调整子图间距并展示所有图像
# Plot all subplots.
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()

# 绘制逻辑回归训练过程中每个类的成本函数曲线
# 训练情况
labels = logistic_regression.unique_labels
for index, label in enumerate(labels):
    plt.plot(range(len(costs[index])), costs[index], label=labels[index])

plt.xlabel('Gradient Steps')
plt.ylabel('Cost')
plt.legend()
plt.show()

# 计算模型在训练集和测试集上的精度
# 测试结果
y_train_predictions = logistic_regression.predict(x_train)
y_test_predictions = logistic_regression.predict(x_test)

train_precision = np.sum(y_train_predictions == y_train) / y_train.shape[0] * 100
test_precision = np.sum(y_test_predictions == y_test) / y_test.shape[0] * 100

# 输出精度结果
print('Training Precision: {:5.4f}%'.format(train_precision))
print('Test Precision: {:5.4f}%'.format(test_precision))

# 设置展示的数字数量和子图大小
# How many numbers to display.
numbers_to_display = 64

# Calculate the number of cells that will hold all the numbers.
num_cells = math.ceil(math.sqrt(numbers_to_display))

# Make the plot a little bit bigger than default one.
plt.figure(figsize=(15, 15))

# 循环展示测试集中的数字及其预测标签
# Go through the first numbers in a test set and plot them.
for plot_index in range(numbers_to_display):
    # 提取当前数字的标签和像素数据
    # Extrace digit data.
    digit_label = y_test[plot_index, 0]
    digit_pixels = x_test[plot_index, :]
    
    # 提取模型对该数字的预测标签
    # Predicted label.
    predicted_label = y_test_predictions[plot_index][0]

    # 根据预测是否正确选择颜色
    color_map = 'Greens' if predicted_label == digit_label else 'Reds'

    # 计算图像大小并重塑像素数据为图像格式
    # Calculate image size (remember that each picture has square proportions).
    image_size = int(math.sqrt(digit_pixels.shape[0]))
    
    frame = digit_pixels.reshape((image_size, image_size))

    # 在子图中展示数字图像及其预测标签
    # Plot the number matrix.
    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(frame, cmap=color_map)
    plt.title(predicted_label)
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# 调整子图间距并展示所有图像
# Plot all subplots.
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()
