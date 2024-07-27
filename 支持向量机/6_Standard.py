from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 划分数据集为训练集和测试集，测试集比例为30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 对训练集和测试集进行标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建SVM模型，使用径向基函数（RBF）作为核函数
svm = SVC(kernel='rbf', C=1, gamma=0.1, random_state=42)
# 在训练集上训练模型
svm.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = svm.predict(X_test)

# 输出分类报告，展示模型的精度、召回率和F1值
print("Classification Report (Standardized Data):")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 绘制混淆矩阵，可视化展示模型的预测结果
print("Confusion Matrix (Standardized Data):")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 输出模型的准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (Standardized Data): {accuracy:.2f}")
