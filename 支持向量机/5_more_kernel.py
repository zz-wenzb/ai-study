from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义不同核函数的 SVM 模型
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# 用于存储不同核函数的分类准确率
results = {}

# 遍历每种核函数，训练SVM模型并评估性能
for kernel in kernels:
    # 初始化支持向量机模型，指定核函数
    svm = SVC(kernel=kernel, C=1, random_state=42)
    # 在训练集上拟合模型
    svm.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = svm.predict(X_test)
    # 计算预测的准确率
    accuracy = accuracy_score(y_test, y_pred)
    # 将准确率存储到结果字典中
    results[kernel] = accuracy
    # 输出分类报告
    print(f"\nClassification Report ({kernel} Kernel):")
    print(classification_report(y_test, y_pred, target_names=iris.target_names, zero_division=0))

    # 绘制混淆矩阵热力图
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({kernel} Kernel)")
    plt.show()

# 绘制不同核函数的准确率比较条形图
plt.bar(results.keys(), results.values())
plt.xlabel("Kernel")
plt.ylabel("Accuracy Score")
plt.title("Accuracy Comparison (Different SVM Kernels)")
plt.show()
