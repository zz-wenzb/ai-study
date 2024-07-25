"""Normalize features"""

import numpy as np


def normalize(features):
    """
    对特征数据进行标准化处理。

    标准化使得数据集中的每个特征具有零均值和单位标准差，这有助于在后续的机器学习算法中减少特征之间的偏差影响。

    参数:
    features: numpy数组，包含待标准化的特征数据。

    返回值:
    features_normalized: 标准化后的特征数据。
    features_mean: 特征数据的均值。
    features_deviation: 特征数据的标准差。
    """

    # 创建一个与输入特征相同形状的浮点数数组，用于存储标准化后的特征
    features_normalized = np.copy(features).astype(float)

    # 计算每个特征的均值
    # 计算均值
    features_mean = np.mean(features, 0)

    # 计算每个特征的标准差
    # 计算标准差
    features_deviation = np.std(features, 0)

    # 如果特征数据包含多于一个样本，那么从中减去均值
    # 标准化操作
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # 将所有标准差为零的特征值设置为1，以避免除以零的错误
    # 防止除以0
    features_deviation[features_deviation == 0] = 1

    # 分母为标准差，如果为零则已提前修正为1，因此这里不会发生除以零的错误
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation
