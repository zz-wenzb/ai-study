"""Prepares the dataset for training"""

import numpy as np
from .normalize import normalize
from .generate_sinusoids import generate_sinusoids
from .generate_polynomials import generate_polynomials


# 准备数据以供训练使用，包括数据的标准化、添加 sinusoidal 特征和多项式特征
def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    """
    对给定的数据进行预处理，包括标准化、添加 sinusoidal 特征和多项式特征。
    这样做的目的是为了提高模型的训练效果和泛化能力。

    参数:
    data: 待处理的数据集，numpy 数组形式。
    polynomial_degree: 多项式特征的最高次数，默认为0，表示不添加多项式特征。
    sinusoid_degree: Sinusoidal 特征的最高次数，默认为0，表示不添加 sinusoidal 特征。
    normalize_data: 是否对数据进行标准化，默认为True。

    返回值:
    data_processed: 处理后的数据集，包括添加的特征。
    features_mean: 标准化数据的均值，用于反向标准化。
    features_deviation: 标准化数据的标准差，用于反向标准化。
    """
    # 获取数据集中的样本数量
    # 计算样本总数
    num_examples = data.shape[0]

    # 复制原始数据，以保留原始数据集不变
    data_processed = np.copy(data)

    # 初始化用于标准化的数据统计量
    # 预处理
    features_mean = 0
    features_deviation = 0

    # 如果需要，对数据进行标准化
    if normalize_data:
        # 标准化数据，并更新数据统计量
        (
            data_normalized,
            features_mean,
            features_deviation
        ) = normalize(data_processed)

        # 更新处理后的数据为标准化后的数据
        data_processed = data_normalized

    # 如果指定，添加 sinusoidal 特征
    # 特征变换sinusoidal
    if sinusoid_degree > 0:
        # 为数据添加 sinusoidal 特征
        sinusoids = generate_sinusoids(data_normalized, sinusoid_degree)
        # 将 sinusoidal 特征与原始数据合并
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)

    # 如果指定，添加多项式特征
    # 特征变换polynomial
    if polynomial_degree > 0:
        # 为数据添加多项式特征
        polynomials = generate_polynomials(data_normalized, polynomial_degree, normalize_data)
        # 将多项式特征与原始数据合并
        data_processed = np.concatenate((data_processed, polynomials), axis=1)

    # 在数据集的最左边添加一列1，作为模型的偏置项
    # 加一列1
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))

    # 返回处理后的数据集，以及用于反向标准化的数据统计量
    return data_processed, features_mean, features_deviation
