# 导入StandardScaler类，用于数据标准化处理
from sklearn.preprocessing import StandardScaler
# 导入numpy库，用于创建和操作数组
import numpy as np
# 导入train_test_split函数，用于将数据集划分为训练集和测试集
from sklearn.model_selection import train_test_split

# 初始化StandardScaler对象，用于后续的数据标准化
scaler = StandardScaler()

# 创建一个5行3列的numpy数组，模拟数据集
X = np.arange(15).reshape(5,3)

# 将数据集划分为训练集和测试集，测试集比例为33%
X_train, X_test = train_test_split(X, test_size=0.33, random_state=42)

# 使用训练集数据拟合标准化器
scaler.fit(X_train)

# 输出标准化器的缩放因子（scale_）、均值（mean_）、方差（var_）和已看到的样本数（n_samples_seen_）
print(scaler.scale_)
print(scaler.mean_)
print(scaler.var_)
print(scaler.n_samples_seen_)

# 对训练集进行标准化处理
print(scaler.transform(X_train))

# 对测试集进行标准化处理，确保与训练集使用相同的标准化参数
print(scaler.transform(X_test))
