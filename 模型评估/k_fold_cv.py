from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 创建并训练模型
model = LogisticRegression(max_iter=1000)

# 使用K折交叉验证评估模型性能
k = 5  # 设置K值
scores = cross_val_score(model, X, y, cv=k)

# 输出每次交叉验证的准确率
print("Accuracy scores for each fold:", scores)

# 计算平均准确率
mean_accuracy = scores.mean()
print("Mean accuracy:", mean_accuracy)
