import pandas as pd
import numpy as np
from math import log

# 定义计算熵的函数
def entropy(data):
    """
    计算给定数据集的熵。

    参数:
    data: pandas.DataFrame, 包含所有特征和标签的数据集。

    返回:
    float, 数据集的熵值。
    """
    labels_count = data['class'].value_counts()
    entropy = -np.sum([(p / len(data)) * log(p / len(data), 2) for p in labels_count])
    return entropy

# 定义计算信息增益的函数
def information_gain(data, feature):
    """
    计算给定特征的信息增益。

    参数:
    data: pandas.DataFrame, 包含所有特征和标签的数据集。
    feature: str, 用于计算信息增益的特征名称。

    返回:
    float, 特征的信息增益值。
    """
    total_entropy = entropy(data)
    weighted_entropy = 0
    for value in data[feature].unique():
        subset = data[data[feature] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)
    gain = total_entropy - weighted_entropy
    return gain

# 定义计算分裂信息的函数
def split_info(data, feature):
    """
    计算给定特征的分裂信息。

    参数:
    data: pandas.DataFrame, 包含所有特征和标签的数据集。
    feature: str, 用于计算分裂信息的特征名称。

    返回:
    float, 特征的分裂信息值。
    """
    split_info = 0
    for value in data[feature].unique():
        p = len(data[data[feature] == value]) / len(data)
        split_info -= p * log(p, 2)
    return split_info

# 定义计算信息增益率的函数
def gain_ratio(data, feature):
    """
    计算给定特征的信息增益率。

    参数:
    data: pandas.DataFrame, 包含所有特征和标签的数据集。
    feature: str, 用于计算信息增益率的特征名称。

    返回:
    float, 特征的信息增益率值。
    """
    gain = information_gain(data, feature)
    split = split_info(data, feature)
    if split == 0:
        return 0
    return gain / split

# 示例数据集
data = pd.DataFrame({
    'outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny',
                'overcast', 'overcast', 'rainy'],
    'temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot',
                    'mild'],
    'humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'normal', 'normal',
                 'high', 'normal', 'high'],
    'wind': ['weak', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong',
             'weak', 'strong'],
    'class': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
})

# 计算每个特征的信息增益率
features = ['outlook', 'temperature', 'humidity', 'wind']
ratios = {feature: gain_ratio(data, feature) for feature in features}

# 找到信息增益率最高的特征
best_feature = max(ratios, key=ratios.get)

print(f"Best feature based on Gain Ratio: {best_feature}")
