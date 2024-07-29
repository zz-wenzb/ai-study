from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 加载手写数字数据集
digits = datasets.load_digits()
# 数据集的特征
X = digits.data
# 数据集的目标标签
y = digits.target

# 对特征数据进行标准化处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 将数据集划分为训练集和测试集，测试集比例为30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用 GridSearchCV 进行超参数调优
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}
# 初始化SVM模型
svm = SVC(random_state=42)
# 使用GridSearchCV进行超参数搜索
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy')
# 在训练集上拟合模型
grid_search.fit(X_train, y_train)

# 输出搜索到的最佳超参数组合
print("Best Parameters:", grid_search.best_params_)

# 获取搜索后的最佳模型
best_svm = grid_search.best_estimator_
# 在测试集上进行预测
y_pred = best_svm.predict(X_test)

# 输出分类报告
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

# 计算预测的准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[str(i) for i in range(10)],
            yticklabels=[str(i) for i in range(10)])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (MNIST Handwritten Digits Recognition)")
plt.show()
