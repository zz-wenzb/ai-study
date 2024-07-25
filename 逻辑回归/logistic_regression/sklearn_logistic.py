# 导入必要的库，用于数据划分、逻辑回归模型构建及模型评估
# 导入必要的库
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import datasets

# 加载iris数据集
# 加载数据集
iris = datasets.load_iris()
# X 代表特征数据
X = iris.data
# y 代表目标标签
y = iris.target

# 将数据集划分为训练集和测试集，测试集比例为30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 创建逻辑回归模型实例
logreg = LogisticRegression()
# 使用训练集训练模型
logreg.fit(X_train, y_train)
# 使用测试集进行预测
y_pred = logreg.predict(X_test)

# 计算预测的准确率
accuracy = accuracy_score(y_test, y_pred)
# 输出准确率
print(f'Accuracy: {accuracy * 100:.2f}%')
