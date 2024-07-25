"""Add polynomial features to the features set"""

import numpy as np
from .normalize import normalize


def generate_polynomials(dataset, polynomial_degree, normalize_data=False):
    """
    生成多项式特征组合。

    根据给定的数据集和多项式次数，生成包含多项式特征的矩阵。
    如果需要，还可以对生成的多项式特征进行标准化处理。

    参数:
    dataset: numpy数组，输入的数据集，包含两列或多列。
    polynomial_degree: 整数，指定生成多项式的最高次数。
    normalize_data: 布尔值，指定是否对生成的多项式特征进行标准化。

    返回:
    numpy数组，包含生成的多项式特征的矩阵。
    """

    # 将数据集按列分割为两个数组
    features_split = np.array_split(dataset, 2, axis=1)
    dataset_1 = features_split[0]
    dataset_2 = features_split[1]

    # 获取每个数据集的样本数和特征数
    (num_examples_1, num_features_1) = dataset_1.shape
    (num_examples_2, num_features_2) = dataset_2.shape

    # 检查两个数据集的样本数是否相同
    if num_examples_1 != num_examples_2:
        raise ValueError('Can not generate polynomials for two sets with different number of rows')

    # 检查数据集是否为空
    if num_features_1 == 0 and num_features_2 == 0:
        raise ValueError('Can not generate polynomials for two sets with no columns')

    # 当其中一个数据集为空时，将其替换为非空的数据集
    if num_features_1 == 0:
        dataset_1 = dataset_2
    elif num_features_2 == 0:
        dataset_2 = dataset_1

    # 确定保留的特征数
    num_features = num_features_1 if num_features_1 < num_examples_2 else num_features_2
    # 根据确定的特征数修剪数据集
    dataset_1 = dataset_1[:, :num_features]
    dataset_2 = dataset_2[:, :num_features]

    # 初始化用于存储多项式特征的矩阵
    polynomials = np.empty((num_examples_1, 0))

    # 生成多项式特征
    for i in range(1, polynomial_degree + 1):
        for j in range(i + 1):
            # 计算当前组合的多项式特征
            polynomial_feature = (dataset_1 ** (i - j)) * (dataset_2 ** j)
            # 将当前多项式特征连接到总特征矩阵中
            polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)

    # 如果需要，对生成的多项式特征进行标准化处理
    if normalize_data:
        polynomials = normalize(polynomials)[0]

    # 返回生成的多项式特征矩阵
    return polynomials
