from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = datasets.load_iris()
# 特征数据
X = iris.data
# 目标标签
y = iris.target

# 将数据集划分为训练集和测试集，测试集比例为30%，随机种子为42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义SVM的参数网格，用于后续的网格搜索
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# 初始化SVM模型，并使用GridSearchCV进行参数搜索
svm = SVC(random_state=42)
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy')
# 在训练集上拟合模型并搜索最佳参数
grid_search.fit(X_train, y_train)

# 输出搜索到的最佳参数组合
print("Best Parameters:", grid_search.best_params_)

# 使用最佳参数重新训练模型，并对测试集进行预测
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)

# 输出分类报告，包含精度、召回率、F1值和支撑度
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 计算预测的准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 绘制混淆矩阵，用于可视化分类结果的准确性
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
