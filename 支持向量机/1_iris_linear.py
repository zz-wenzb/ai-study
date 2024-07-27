from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = datasets.load_iris()
# 数据集的特征
X = iris.data
# 数据集的目标标签
y = iris.target

# 将数据集划分为训练集和测试集，测试集占30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建一个线性核函数的SVM分类器
svm = SVC(kernel='linear', C=1, random_state=42)
# 使用训练集对分类器进行训练
svm.fit(X_train, y_train)

# 使用训练好的分类器对测试集进行预测
y_pred = svm.predict(X_test)

# 打印分类报告，包含精度、召回率、F1值等指标
# 输出分类报告和混淆矩阵
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 绘制混淆矩阵热力图，用于直观显示分类器的预测效果
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 计算并打印分类器的准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
