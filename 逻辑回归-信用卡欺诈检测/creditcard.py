import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 忽略警告信息，以减少输出干扰
warnings.filterwarnings('ignore')

# 读取信用卡欺诈数据集
data = pd.read_csv("creditcard.csv")

# 显示数据集的前五行，以检查数据加载是否正确
# print(data.head())

# 计算并排序'Class'列中每个类的值的数量
count_classes = pd.value_counts(data['Class'], sort=True).sort_index()

# 绘制每个类的值的数量的柱状图
count_classes.plot(kind='bar')
plt.title("Fraud class histogram")  # 设置图表标题
plt.xlabel("Class")  # 设置x轴标签
plt.ylabel("Frequency")  # 设置y轴标签
plt.show()  # 显示图表

# 使用StandardScaler对'Amount'列进行标准化处理，使数据在特征尺度上标准化
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

# 从数据集中删除'Time'和'Amount'列，因为它们可能与标准化后的数据冗余或不相关
data = data.drop(['Time', 'Amount'], axis=1)

# 打印处理后的数据集的前五行，以验证数据的正确性和完整性
print(data.head())

# 打印所有类别为1的数据的索引，这可能用于异常值检测或不平衡数据的分析
print(data[data.Class == 1].index)

# 从数据集中分离特征和标签
# 将所有不是'Class'列的数据作为特征X
X = data.iloc[:, data.columns != 'Class']
# 将'Class'列的数据作为标签y
y = data.iloc[:, data.columns == 'Class']

# 获取所有异常样本的索引
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

# 获取所有正常样本的索引
normal_indices = data[data.Class == 0].index

# 从正常样本中随机选择与异常样本数量相同数量的样本索引
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
# 确保选择的正常样本索引是numpy数组格式
random_normal_indices = np.array(random_normal_indices)

# 将欺诈和随机选取的正常样本索引合并，实现数据的欠采样
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

# 通过索引获取欠采样后的所有样本数据
under_sample_data = data.iloc[under_sample_indices, :]

# 分离特征和标签，特征为除'Class'列外的所有列
X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
# 标签为'Class'列
y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']

# 下采样 样本比例
print("正常样本所占整体比例: ", len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data))
print("异常样本所占整体比例: ", len(under_sample_data[under_sample_data.Class == 1]) / len(under_sample_data))
print("下采样策略总体样本数量: ", len(under_sample_data))

# 整个数据集进行划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("原始训练集包含样本数量: ", len(X_train))
print("原始测试集包含样本数量: ", len(X_test))
print("原始样本总数: ", len(X_train) + len(X_test))

# 下采样数据集进行划分
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample
                                                                                                    , y_undersample
                                                                                                    , test_size=0.3
                                                                                                    , random_state=0)
print("")
print("下采样训练集包含样本数量: ", len(X_train_undersample))
print("下采样测试集包含样本数量: ", len(X_test_undersample))
print("下采样样本总数: ", len(X_train_undersample) + len(X_test_undersample))


# Recall = TP/(TP+FN)

def printing_Kfold_scores(x_train_data, y_train_data):
    """
    使用K折交叉验证打印不同正则化参数下的平均召回率

    参数:
    x_train_data: 训练集的特征数据
    y_train_data: 训练集的目标数据

    返回:
    best_c: 表现最好的正则化参数
    """
    # 初始化5折交叉验证对象，不打乱数据顺序
    fold = KFold(5, shuffle=False)

    # 定义不同力度的正则化惩罚力度
    c_param_range = [0.01, 0.1, 1, 10, 100]
    # 展示结果用的表格
    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range

    # 循环遍历不同的正则化参数
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('正则化惩罚力度: ', c_param)
        print('-------------------------------------------')
        print('')

        # 保存每次折叠的召回率
        recall_accs = []
        j = 0
        # 执行K折交叉验证
        for iteration, indices in enumerate(fold.split(x_train_data)):
            # 使用逻辑回归模型，指定正则化参数
            lr = LogisticRegression(C=c_param, penalty='l1', solver='liblinear')

            # 使用训练集拟合模型
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())

            # 使用验证集进行预测
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)

            # 计算并打印本次折叠的召回率
            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration ', iteration, ': 召回率 = ', recall_acc)

        # 计算并保存平均召回率
        results_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('平均召回率 ', np.mean(recall_accs))
        print('')

    # 选取平均召回率最高的正则化参数
    best_c = results_table.loc[results_table['Mean recall score'].astype('float32').idxmax()]['C_parameter']

    # 打印最佳参数
    print('*********************************************************************************')
    print('效果最好的模型所选参数 = ', best_c)
    print('*********************************************************************************')

    return best_c


best_c = printing_Kfold_scores(X_train_undersample, y_train_undersample)


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    绘制混淆矩阵
    """
    # 使用热图绘制混淆矩阵，参数包括混淆矩阵数据、标题和颜色映射
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)  # 设置图表标题
    plt.colorbar()  # 添加颜色条
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)  # 设置x轴标签
    plt.yticks(tick_marks, classes)  # 设置y轴标签

    # 根据混淆矩阵的最大值设定文本颜色的阈值
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # 在每个格子上标注对应的数值
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # 调整布局，避免子图之间的重叠
    plt.tight_layout()
    plt.ylabel('True label')  # 设置y轴标题
    plt.xlabel('Predicted label')  # 设置x轴标题


import itertools

# 使用最佳的C值初始化带有L1惩罚项的逻辑回归模型
lr = LogisticRegression(C=best_c, penalty='l1', solver='liblinear')
# 在欠采样的训练数据上训练模型
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
# 使用测试数据进行预测
y_pred_undersample = lr.predict(X_test_undersample.values)

# 计算混淆矩阵以评估模型性能
cnf_matrix = confusion_matrix(y_test_undersample, y_pred_undersample)
# 设置打印选项，保留两位小数
np.set_printoptions(precision=2)

# 计算并打印模型的召回率
print("召回率: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

# 设置绘图参数
class_names = [0, 1]
plt.figure()
# 绘制混淆矩阵
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
# 显示图形
plt.show()

# 使用最佳的C值和L1正则化创建逻辑回归模型
lr = LogisticRegression(C=best_c, penalty='l1', solver='liblinear')
# 在欠采样的训练数据上训练模型
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
# 在测试数据上进行预测
y_pred = lr.predict(X_test.values)

# 计算混淆矩阵
cnf_matrix = confusion_matrix(y_test, y_pred)
# 设置打印选项，保留两位小数
np.set_printoptions(precision=2)

# 计算并打印召回率
print("召回率: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

# 设置绘图的类别名称
class_names = [0, 1]
# 创建一个新的图形窗口
plt.figure()
# 绘制混淆矩阵
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
# 显示图形
plt.show()

# 使用最佳参数创建逻辑回归模型实例
lr = LogisticRegression(C=0.01, penalty='l1', solver='liblinear')

# 使用欠采样的训练数据拟合模型
lr.fit(X_train_undersample, y_train_undersample.values.ravel())

# 预测测试集的概率值
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)

# 定义不同的预测阈值列表
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 创建一个图形框，用于绘制不同阈值下的混淆矩阵
plt.figure(figsize=(10, 10))

# 初始化子图的计数器
j = 1

# 遍历不同的阈值，绘制混淆矩阵
for i in thresholds:
    # 根据当前阈值获取预测结果
    y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > i

    # 创建3x3子图网格中的下一个子图
    plt.subplot(3, 3, j)
    j += 1

    # 计算混淆矩阵
    cnf_matrix = confusion_matrix(y_test_undersample, y_test_predictions_high_recall)
    np.set_printoptions(precision=2)

    # 打印给定阈值下的测试集召回率
    print("给定阈值为:", i, "时测试集召回率: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

    # 定义类别名称
    class_names = [0, 1]
    # 绘制混淆矩阵
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Threshold >= %s' % i)

# 读取信用卡欺诈数据集
credit_cards = pd.read_csv('creditcard.csv')

# 获取数据集的列名
columns = credit_cards.columns
# 在特征中去除掉标签列，得到特征列名
features_columns = columns.delete(len(columns) - 1)

# 根据列名提取特征和标签
features = credit_cards[features_columns]
labels = credit_cards['Class']

# 将数据集分割为训练集和测试集
features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                            labels,
                                                                            test_size=0.3,
                                                                            random_state=0)

# 使用SMOTE过采样技术平衡训练集中的数据
oversampler = SMOTE(random_state=0)
os_features, os_labels = oversampler.fit_resample(features_train, labels_train)

# 打印过采样后，标签为1的数量
print(len(os_labels[os_labels == 1]))

# 将过采样后的特征和标签转换为DataFrame格式
os_features = pd.DataFrame(os_features)
os_labels = pd.DataFrame(os_labels)
# 通过K折交叉验证找到最佳的C值
best_c = printing_Kfold_scores(os_features, os_labels)

# 使用最佳C值初始化逻辑回归模型，并使用L1正则化
lr = LogisticRegression(C=best_c, penalty='l1', solver='liblinear')
# 训练模型
lr.fit(os_features, os_labels.values.ravel())
# 预测测试集的结果
y_pred = lr.predict(features_test.values)

# 计算混淆矩阵
cnf_matrix = confusion_matrix(labels_test, y_pred)
# 设置打印选项，保留两位小数
np.set_printoptions(precision=2)

# 计算并打印召回率
print("召回率: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

# 绘制混淆矩阵
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()
