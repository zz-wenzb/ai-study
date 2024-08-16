import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class AdaBoost:
    """
    AdaBoost算法的实现。

    参数:
    - n_estimators: 弱分类器的数量，默认为50。
    - learning_rate: 学习率，用于调整弱分类器的权重，默认为1.0。

    该类通过迭代的方式训练多个弱分类器，并将它们组合成一个强分类器。
    """
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators  # 弱分类器的数量
        self.learning_rate = learning_rate  # 学习率
        self.classifiers = []  # 存储训练好的弱分类器
        self.classifier_weights = []  # 存储每个弱分类器的权重

    def fit(self, X, y):
        """
        训练AdaBoost模型。

        参数:
        - X: 输入特征，形状为(n_samples, n_features)。
        - y: 目标标签，形状为(n_samples,)。

        该方法通过迭代训练多个决策树分类器，并更新每个样本的权重。
        """
        n_samples, _ = X.shape
        sample_weights = np.ones(n_samples) / n_samples  # 初始化样本权重

        for _ in range(self.n_estimators):
            classifier = DecisionTreeClassifier(max_depth=1)  # 创建一个弱分类器
            classifier.fit(X, y, sample_weight=sample_weights)  # 使用带有样本权重的训练数据拟合分类器

            # 计算错误率
            predictions = classifier.predict(X)
            incorrect = (predictions != y)
            error = np.sum(sample_weights[incorrect]) / np.sum(sample_weights)

            # 计算分类器的权重
            classifier_weight = 0.5 * np.log((1.0 - error) / (error + 1e-10))

            # 更新样本权重
            sample_weights *= np.exp(-classifier_weight * y * predictions)
            sample_weights /= np.sum(sample_weights)

            # 存储分类器及其权重
            self.classifiers.append(classifier)
            self.classifier_weights.append(classifier_weight)

    def predict(self, X):
        """
        使用AdaBoost模型进行预测。

        参数:
        - X: 输入特征，形状为(n_samples, n_features)。

        返回:
        - 预测的标签，形状为(n_samples,)。
        """
        predictions = [classifier.predict(X) for classifier in self.classifiers]  # 每个弱分类器的预测结果
        weighted_predictions = np.array(predictions) * np.array(self.classifier_weights)[:, None]  # 加权预测结果
        return np.sign(np.sum(weighted_predictions, axis=0))  # 返回加权预测结果的符号作为最终预测

# 创建一个随机数据集
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练AdaBoost模型
model = AdaBoost(n_estimators=50)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
