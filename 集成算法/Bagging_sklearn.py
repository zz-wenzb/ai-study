# 导入所需的库
from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义基础分类器
base_estimator = DecisionTreeClassifier()

# 创建Bagging分类器
bagging = BaggingClassifier(estimator=base_estimator, n_estimators=10, max_samples=0.8, random_state=42)

# 拟合模型
bagging.fit(X_train, y_train)

# 预测
y_pred = bagging.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
