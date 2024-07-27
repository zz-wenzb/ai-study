from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn import datasets
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义不同机器学习模型的字典
models = {
    "SVM": SVC(kernel='rbf', C=1, gamma=0.1, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNeighbors": KNeighborsClassifier(n_neighbors=5)
}

# 训练模型并计算准确率
accuracy_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = accuracy
    # 输出分类报告
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 绘制不同模型准确率的对比图
plt.bar(accuracy_scores.keys(), accuracy_scores.values())
plt.xlabel("Model")
plt.ylabel("Accuracy Score")
plt.title("Model Comparison on Iris Dataset")
plt.show()
