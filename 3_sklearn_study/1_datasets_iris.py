# 导入sklearn.datasets模块的load_iris函数，用于加载iris数据集
from sklearn.datasets import load_iris

# 加载iris数据集
iris = load_iris()

# 输出目标变量的值，这些值代表了三种不同类型的鸢尾花
# print(iris.data)
print(iris.target)

# 输出数据集的形状，即样本数量和特征数量
print(iris.data.shape)

# 输出目标变量的形状，即样本数量
print(iris.target.shape)

# 输出特征名称，这些名称描述了每个特征的物理意义
print(iris.feature_names)

# 输出目标变量的名称，这些名称对应于三种不同的鸢尾花类型
print(iris.target_names)

print("================================")

# 输出数据集的描述，包括数据集的来源、特征信息和目标变量信息
print(iris.DESCR)

# 输出第一个样本的特征值，这些值描述了该样本的物理特性
print(iris.data[0])

# 输出第一个样本的目标变量值，即该样本属于哪种类型的鸢尾花
print(iris.target[0])

# 同时输出第一个样本的特征值和目标变量值，展示数据集中的一条完整记录
print(iris.data[0], iris.target[0])