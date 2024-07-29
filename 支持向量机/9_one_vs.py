from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = datasets.load_iris()
# 特征数据
X = iris.data
# 目标标签
y = iris.target

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建支持向量机模型（One-vs-One 策略）
svm_ovo = SVC(kernel='rbf', C=1, gamma=0.1, decision_function_shape='ovo', random_state=42)
# 在训练集上训练模型
svm_ovo.fit(X_train, y_train)
# 在测试集上进行预测
y_pred_ovo = svm_ovo.predict(X_test)

# 输出One-vs-One策略的分类报告和混淆矩阵
print("\nClassification Report (One-vs-One):")
print(classification_report(y_test, y_pred_ovo, target_names=iris.target_names))

print("\nConfusion Matrix (One-vs-One):")
cm_ovo = confusion_matrix(y_test, y_pred_ovo)
sns.heatmap(cm_ovo, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (One-vs-One)")
plt.show()

# 创建支持向量机模型（One-vs-Rest 策略）
svm_ovr = SVC(kernel='rbf', C=1, gamma=0.1, decision_function_shape='ovr', random_state=42)
# 在训练集上训练模型
svm_ovr.fit(X_train, y_train)
# 在测试集上进行预测
y_pred_ovr = svm_ovr.predict(X_test)

# 输出One-vs-Rest策略的分类报告和混淆矩阵
print("\nClassification Report (One-vs-Rest):")
print(classification_report(y_test, y_pred_ovr, target_names=iris.target_names))

print("\nConfusion Matrix (One-vs-Rest):")
cm_ovr = confusion_matrix(y_test, y_pred_ovr)
sns.heatmap(cm_ovr, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (One-vs-Rest)")
plt.show()
