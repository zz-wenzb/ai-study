import numpy as np


def generate_sinusoids(dataset, sinusoid_degree):
    """
    根据给定的输入数据集和正弦波度数，生成一个包含多个正弦波特征的矩阵。

    参数:
    dataset: 输入数据集，一个numpy数组。
    sinusoid_degree: 生成正弦波的度数，决定生成正弦波的种类数量。

    返回:
    一个numpy数组，包含所有生成的正弦波特征。
    """
    """
    sin(x).
    """
    # 获取输入数据集的样本数量
    num_examples = dataset.shape[0]
    # 初始化一个空数组，用于存储所有正弦波特征
    sinusoids = np.empty((num_examples, 0))

    # 遍历指定的度数范围，生成每个度数对应的正弦波特征
    for degree in range(1, sinusoid_degree + 1):
        # 计算当前度数的正弦波特征
        sinusoid_features = np.sin(degree * dataset)
        # 将当前度数的正弦波特征并入到总的正弦波特征数组中
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)

    # 返回包含所有正弦波特征的数组
    return sinusoids
