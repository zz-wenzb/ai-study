import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义基础分类器的数量
n_estimators = 10

# 创建一个列表来存储所有的基础分类器
estimators = []

# 实现Bagging
for i in range(n_estimators):
    # 有放回地从训练集中抽取样本
    bootstrap_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_bootstrap, y_bootstrap = X_train[bootstrap_idx], y_train[bootstrap_idx]

    # 创建并训练决策树
    tree = DecisionTreeClassifier(random_state=i)
    tree.fit(X_bootstrap, y_bootstrap)

    # 将训练好的决策树添加到列表中
    estimators.append(tree)

# 使用所有的基础分类器进行预测
y_preds = []
for estimator in estimators:
    y_pred = estimator.predict(X_test)
    y_preds.append(y_pred)

# 通过投票确定最终的预测结果
final_predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=np.array(y_preds))

# 计算准确率
accuracy = accuracy_score(y_test, final_predictions)
print(f"Accuracy: {accuracy}")
