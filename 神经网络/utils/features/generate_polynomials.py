"""为特征集添加多项式特征"""

import numpy as np
from .normalize import normalize


def generate_polynomials(dataset, polynomial_degree, normalize_data=False):
    """
    生成多项式特征

    参数:
        dataset (ndarray): 输入数据集，应为二维数组，其中每一列代表一个特征。
        polynomial_degree (int): 多项式的最高次数。
        normalize_data (bool): 是否对生成的多项式特征进行归一化，默认为 False。

    返回:
        ndarray: 包含多项式特征的新数据集。
    """

    # 将输入数据集按列分为两部分
    features_split = np.array_split(dataset, 2, axis=1)
    dataset_1 = features_split[0]
    dataset_2 = features_split[1]

    (num_examples_1, num_features_1) = dataset_1.shape
    (num_examples_2, num_features_2) = dataset_2.shape

    # 检查两个数据集的样本数量是否一致
    if num_examples_1 != num_examples_2:
        raise ValueError('无法为行数不同的两个数据集生成多项式特征')

    # 检查两个数据集是否有至少一个特征
    if num_features_1 == 0 and num_features_2 == 0:
        raise ValueError('无法为无特征的数据集生成多项式特征')

    # 如果其中一个数据集没有特征，则使用另一个数据集
    if num_features_1 == 0:
        dataset_1 = dataset_2
    elif num_features_2 == 0:
        dataset_2 = dataset_1

    # 确定使用的特征数量
    num_features = num_features_1 if num_features_1 <= num_features_2 else num_features_2
    dataset_1 = dataset_1[:, :num_features]
    dataset_2 = dataset_2[:, :num_features]

    # 初始化多项式特征矩阵
    polynomials = np.empty((num_examples_1, 0))

    # 生成多项式特征
    for i in range(1, polynomial_degree + 1):
        for j in range(i + 1):
            polynomial_feature = (dataset_1 ** (i - j)) * (dataset_2 ** j)
            polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)

    # 如果需要，对多项式特征进行归一化处理
    if normalize_data:
        polynomials = normalize(polynomials)[0]

    return polynomials
