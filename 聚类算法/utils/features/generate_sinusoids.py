import numpy as np


def generate_sinusoids(dataset, sinusoid_degree):
    """
    生成包含多个正弦波特征的数组。

    对于给定的数据集，这个函数通过计算不同频率的正弦波并与原始数据集拼接，
    从而扩展数据集的特征维度。每个新特征是原始数据集上某个度数的正弦函数。

    参数:
    dataset: numpy数组，输入的数据集，它是一维的。
    sinusoid_degree: 整数，表示要生成的正弦波的最大度数。

    返回:
    numpy数组，包含了原始数据集和多个正弦波特征的组合。
    """
    """
    sin(x).
    """
    # 获取数据集中的样本数量
    num_examples = dataset.shape[0]
    # 初始化一个空数组，用于存储所有正弦波特征
    sinusoids = np.empty((num_examples, 0))

    # 遍历指定的正弦波度数，从1到sinusoid_degree
    for degree in range(1, sinusoid_degree + 1):
        # 计算当前度数的正弦波特征，并将其追加到sinusoids数组中
        sinusoid_features = np.sin(degree * dataset)
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)

    return sinusoids
