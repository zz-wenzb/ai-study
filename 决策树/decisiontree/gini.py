import pandas as pd
from collections import Counter


# 计算基尼指数的函数
def gini_index(groups, classes):
    # 计算总的样本数
    n_instances = float(sum([len(group) for group in groups]))
    # 总基尼指数初始化为0
    gini = 0.0

    for group in groups:
        size = float(len(group))
        # 避免除以零
        if size == 0:
            continue
        score = 0.0
        # 统计类别数量
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p ** 2
        # 加权基尼指数
        gini += (1.0 - score) * (size / n_instances)

    return gini


# 选择最佳分割点的函数
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 1, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = split_dataset(dataset, index, row[index])
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# 分割数据集的函数
def split_dataset(dataset, index, value):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# 创建决策树的函数
def create_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, train, max_depth, min_size, 1)
    return root


# 递归地构建子树
def split(node, train, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])

    if not left or not right:  # 添加检查以确保左右子集都不是空的
        node['left'] = node['right'] = predict_class(train)  # 使用整个训练集预测
    else:
        # 如果左子树为空或大小小于最小大小
        if not left or len(left) <= min_size:
            node['left'] = predict_class(left)
        else:
            node['left'] = get_split(left)
            # 如果未达到最大深度，继续分割左子树
            if depth < max_depth:
                split(node['left'], left, max_depth, min_size, depth + 1)

        # 如果右子树为空或大小小于最小大小
        if not right or len(right) <= min_size:
            node['right'] = predict_class(right)
        else:
            node['right'] = get_split(right)
            # 如果未达到最大深度，继续分割右子树
            if depth < max_depth:
                split(node['right'], right, max_depth, min_size, depth + 1)


# 预测叶节点类别的函数
def predict_class(group):
    if not group:  # 检查group是否为空
        return None
    outcomes = [row[-1] for row in group]
    prediction = max(set(outcomes), key=outcomes.count)
    return prediction


# 打印决策树的函数
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth * ' ', (node['index'] + 1), node['value'])))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * ' ', node)))


# 示例数据集
dataset = [
    ['sunny', 'hot', 'high', 'False'],
    ['sunny', 'hot', 'high', 'False'],
    ['overcast', 'hot', 'high', 'True'],
    ['rain', 'mild', 'high', 'True'],
    ['rain', 'cool', 'normal', 'True'],
    ['rain', 'cool', 'normal', 'True'],
    ['overcast', 'cool', 'normal', 'True'],
    ['sunny', 'mild', 'high', 'False'],
    ['sunny', 'cool', 'normal', 'True'],
    ['rain', 'mild', 'normal', 'True'],
    ['sunny', 'mild', 'normal', 'True'],
    ['overcast', 'mild', 'high', 'True'],
    ['overcast', 'hot', 'normal', 'True'],
    ['rain', 'mild', 'high', 'False']
]

# 将数据集转换为DataFrame
df = pd.DataFrame(dataset, columns=['outlook', 'temperature', 'humidity', 'play'])

# 将分类数据编码为数值
df['outlook'] = df['outlook'].astype('category').cat.codes
df['temperature'] = df['temperature'].astype('category').cat.codes
df['humidity'] = df['humidity'].astype('category').cat.codes
df['play'] = df['play'].astype('category').cat.codes

# 转换回列表格式
dataset = df.values.tolist()

# 构建决策树
tree = create_tree(dataset, max_depth=5, min_size=1)

# 打印决策树
print_tree(tree)
