# 导入机器学习相关的库，用于数据处理、特征提取、模型训练和评估
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 定义感兴趣的新闻组类别，用于数据集的过滤
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'talk.politics.guns']
# 加载20个新闻组数据集的全部数据，并仅保留指定的类别
newsgroups = fetch_20newsgroups(subset='all', categories=categories)

# 使用TF-IDF向量化文本数据，将文本转换为数值特征向量
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data)
# 标记文本所属的类别
y = newsgroups.target

# 将数据集划分为训练集和测试集，用于模型训练和评估
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建一个线性SVM模型，用于文本分类
svm = SVC(kernel='linear', C=1, random_state=42)
# 在训练集上训练模型
svm.fit(X_train, y_train)

# 使用训练好的模型对测试集进行预测
y_pred = svm.predict(X_test)

# 输出分类报告，包括精度、召回率和F1值等指标
print("\nClassification Report (Text Classification):")
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))

# 绘制混淆矩阵，可视化模型的分类效果
print("\nConfusion Matrix (Text Classification):")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=newsgroups.target_names,
            yticklabels=newsgroups.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 计算预测的准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (Text Classification): {accuracy:.2f}")
