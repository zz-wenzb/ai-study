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

# 创建 SVM 模型（使用 RBF 核）
svm_rbf = SVC(kernel='rbf', C=1, gamma=0.1, random_state=42)
# 在训练集上训练模型
svm_rbf.fit(X_train, y_train)

# 预测测试集
y_pred_rbf = svm_rbf.predict(X_test)

# 输出分类报告，包括精度、召回率和F1值
print("\nClassification Report (RBF Kernel):")
print(classification_report(y_test, y_pred_rbf, target_names=iris.target_names))

# 绘制混淆矩阵热力图
print("Confusion Matrix (RBF Kernel):")
cm_rbf = confusion_matrix(y_test, y_pred_rbf)
sns.heatmap(cm_rbf, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 输出模型的准确率
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"Accuracy (RBF Kernel): {accuracy_rbf:.2f}")
