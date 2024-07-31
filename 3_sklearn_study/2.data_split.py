# 导入train_test_split函数，用于将数据集分割为训练集和测试集
from sklearn.model_selection import train_test_split
# 导入iris数据集，用于演示机器学习模型的训练和测试
from sklearn.datasets import load_iris

# 加载iris数据集
iris = load_iris()

# 使用train_test_split将数据集分割为训练集和测试集，测试集比例为30%
# 分割数据集是为了在训练模型后，能在独立的测试集上评估模型的泛化能力
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

# 打印训练集的特征数组形状
print(X_train.shape)
# 打印测试集的特征数组形状
print(X_test.shape)
# 打印训练集的目标数组形状
print(y_train.shape)
# 打印测试集的目标数组形状
print(y_test.shape)
