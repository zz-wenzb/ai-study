import numpy as np
from 线性回归.utils.features import prepare_for_training


class LinearRegression:
    """
    线性回归类，用于训练线性回归模型和进行预测。

    参数:
    data: 训练数据，二维numpy数组。
    labels: 标签值，一维numpy数组。
    polynomial_degree: 多项式拟合的最高次数。
    sinusoid_degree: 正弦波拟合的最高次数。
    normalize_data: 是否对数据进行标准化处理。
    """

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        初始化线性回归对象，包括数据预处理和参数初始化。
        """
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data=True)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))

    def train(self, alpha, num_iterations=500):
        """
        训练线性回归模型。

        参数:
        alpha: 学习率。
        num_iterations: 梯度下降的迭代次数。

        返回:
        theta: 训练得到的模型参数。
        cost_history: 模型在训练过程中损失函数的变化历史。
        """
        """
                    训练模块，执行梯度下降
        """
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        """
        梯度下降算法，用于更新模型参数。

        参数:
        alpha: 学习率。
        num_iterations: 迭代次数。

        返回:
        cost_history: 模型在训练过程中损失函数的变化历史。
        """
        """
                    实际迭代模块，会迭代num_iterations次
        """
        cost_history = []
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def gradient_step(self, alpha):
        """
        梯度下降的单步更新操作。

        参数:
        alpha: 学习率。
        """
        """
                    梯度下降参数更新计算方法，注意是矩阵运算
        """
        num_examples = self.data.shape[0]
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        delta = prediction - self.labels
        theta = self.theta
        theta = theta - alpha * (1 / num_examples) * (np.dot(delta.T, self.data)).T
        self.theta = theta

    def cost_function(self, data, labels):
        """
        计算模型的损失函数。

        参数:
        data: 输入数据。
        labels: 真实标签。

        返回:
        cost: 模型的损失。
        """
        """
                    损失计算方法
        """
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(self.data, self.theta) - labels
        cost = (1 / 2) * np.dot(delta.T, delta) / num_examples
        return cost[0][0]

    @staticmethod
    def hypothesis(data, theta):
        """
        根据模型参数和输入数据，预测输出值。

        参数:
        data: 输入数据。
        theta: 模型参数。

        返回:
        predictions: 预测值。
        """
        predictions = np.dot(data, theta)
        return predictions

    def get_cost(self, data, labels):
        """
        计算给定数据和标签的损失函数值。

        参数:
        data: 输入数据。
        labels: 真实标签。

        返回:
        cost: 给定数据的损失函数值。
        """
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]

        return self.cost_function(data_processed, labels)

    def predict(self, data):
        """
        使用训练好的模型进行预测。

        参数:
        data: 需要预测的输入数据。

        返回:
        predictions: 预测结果。
        """
        """
                    用训练的参数模型，与预测得到回归值结果
        """
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]

        predictions = LinearRegression.hypothesis(data_processed, self.theta)

        return predictions
