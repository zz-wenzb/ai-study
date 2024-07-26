# 导入numpy、pandas和matplotlib.pyplot库，用于数据处理和可视化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 从k_means模块导入KMeans类，用于执行KMeans聚类算法
from k_means import KMeans

# 读取iris数据集
data = pd.read_csv('../data/iris.csv')
# 定义iris的种类名称
iris_types = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']

# 定义x轴和y轴的特征名称
x_axis = 'petal_length'
y_axis = 'petal_width'

# 创建一个1行2列的子图，展示已知标签和未知标签的数据分布
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# 根据种类绘制已知标签的数据点
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class'] == iris_type], data[y_axis][data['class'] == iris_type], label=iris_type)
plt.title('label known')
plt.legend()

plt.subplot(1, 2, 2)
# 绘制所有数据点，此时标签未知
plt.scatter(data[x_axis][:], data[y_axis][:])
plt.title('label unknown')
plt.show()

# 获取数据集中的样本数量
num_examples = data.shape[0]
# 将数据集中的x和y特征提取出来，并重塑为适合算法输入的形状
x_train = data[[x_axis, y_axis]].values.reshape(num_examples, 2)

# 指定聚类的数量和最大迭代次数
# 指定好训练所需的参数
num_clusters = 3
max_iteritions = 50

# 初始化KMeans算法对象
k_means = KMeans(x_train, num_clusters)
# 训练KMeans模型
centroids, closest_centroids_ids = k_means.train(max_iteritions)

# 创建一个1行2列的子图，展示已知标签和KMeans聚类后的数据分布
# 对比结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# 根据种类绘制已知标签的数据点
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class'] == iris_type], data[y_axis][data['class'] == iris_type], label=iris_type)
plt.title('label known')
plt.legend()

plt.subplot(1, 2, 2)
# 根据聚类结果绘制数据点，并标记聚类中心
for centroid_id, centroid in enumerate(centroids):
    current_examples_index = (closest_centroids_ids == centroid_id).flatten()
    plt.scatter(data[x_axis][current_examples_index], data[y_axis][current_examples_index], label=centroid_id)
# 绘制聚类中心点
for centroid_id, centroid in enumerate(centroids):
    plt.scatter(centroid[0], centroid[1], c='black', marker='x')
plt.legend()
plt.title('label kmeans')
plt.show()
