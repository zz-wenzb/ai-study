import numpy as np


def generate_sinusoids(dataset, sinusoid_degree):
    """
    根据给定的数据集和正弦波度数，生成一个包含多个正弦波特征的矩阵。

    参数:
    dataset: numpy数组，输入的数据集，假设是一维的。
    sinusoid_degree: 整数，表示需要生成的正弦波的最大度数。

    返回:
    numpy数组，包含所有生成的正弦波特征的矩阵。
    """
    """
    sin(x).
    """
    # 获取数据集中的例子数量
    num_examples = dataset.shape[0]
    # 初始化一个空的numpy数组，用于存储所有正弦波特征
    sinusoids = np.empty((num_examples, 0))

    # 对于每个度数，从1到给定的最大度数，生成一个正弦波
    for degree in range(1, sinusoid_degree + 1):
        # 生成当前度数的正弦波特征
        sinusoid_features = np.sin(degree * dataset)
        # 将当前度数的正弦波特征并入到存储所有正弦波特征的数组中
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)

    # 返回包含所有正弦波特征的数组
    return sinusoids
