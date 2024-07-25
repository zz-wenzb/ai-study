import numpy as np
from scipy.optimize import minimize
from 逻辑回归.utils.features import prepare_for_training
from 逻辑回归.utils.hypothesis import sigmoid


# 逻辑回归类
class LogisticRegression:
    """
    逻辑回归模型类，用于训练和预测。

    参数:
    data: 训练数据，二维numpy数组。
    labels: 数据标签，一维numpy数组。
    polynomial_degree: 多项式特征的最高次数。
    sinusoid_degree: 正弦波特征的最高次数。
    normalize_data: 是否对数据进行标准化处理。
    """

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=False):
        """
        初始化模型，包括数据预处理和参数初始化。
        """
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data=False)

        self.data = data_processed
        self.labels = labels
        self.unique_labels = np.unique(labels)
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]
        num_unique_labels = np.unique(labels).shape[0]
        self.theta = np.zeros((num_unique_labels, num_features))

    def train(self, max_iterations=1000):
        """
        训练逻辑回归模型。

        参数:
        max_iterations: 最大迭代次数。

        返回:
        训练好的参数theta和每个迭代周期的成本历史记录。
        """
        cost_histories = []
        num_features = self.data.shape[1]
        # enumerate()函数用于在迭代过程中同时获取元素的索引和值。
        for label_index, unique_label in enumerate(self.unique_labels):
            # current_initial_theta = np.copy(self.theta[label_index].reshape(num_features, 1))
            current_initial_theta = np.copy(self.theta[label_index])
            current_lables = (self.labels == unique_label).astype(float)
            (current_theta, cost_history) = LogisticRegression.gradient_descent(self.data, current_lables,
                                                                                current_initial_theta, max_iterations)
            self.theta[label_index] = current_theta.T
            cost_histories.append(cost_history)

        return self.theta, cost_histories

    @staticmethod
    def gradient_descent(data, labels, current_initial_theta, max_iterations):
        """
        使用梯度下降法优化逻辑回归模型的参数。

        参数:
        data: 训练数据。
        labels: 数据标签。
        current_initial_theta: 当前的初始参数向量。
        max_iterations: 最大迭代次数。

        返回:
        最终优化后的参数和成本历史记录。
        """
        cost_history = []
        num_features = data.shape[1]
        result = minimize(
            lambda current_theta: LogisticRegression.cost_function(data, labels,
                                                                   current_theta.reshape(num_features, 1)),
            current_initial_theta,
            method='CG',
            jac=lambda current_theta: LogisticRegression.gradient_step(data, labels,
                                                                       current_theta.reshape(num_features, 1)),
            callback=lambda current_theta: cost_history.append(
                LogisticRegression.cost_function(data, labels, current_theta.reshape((num_features, 1)))),
            options={'maxiter': max_iterations}
        )
        if not result.success:
            raise ArithmeticError('Can not minimize cost function' + result.message)
        optimized_theta = result.x.reshape(num_features, 1)
        return optimized_theta, cost_history

    @staticmethod
    def cost_function(data, labels, theat):
        """
        计算逻辑回归模型的成本函数。

        参数:
        data: 训练数据。
        labels: 数据标签。
        theat: 参数向量。

        返回:
        当前参数下的成本函数值。
        """
        num_examples = data.shape[0]
        predictions = LogisticRegression.hypothesis(data, theat)
        y_is_set_cost = np.dot(labels[labels == 1].T, np.log(predictions[labels == 1]))
        y_is_not_set_cost = np.dot(1 - labels[labels == 0].T, np.log(1 - predictions[labels == 0]))
        cost = (-1 / num_examples) * (y_is_set_cost + y_is_not_set_cost)
        return cost

    @staticmethod
    def hypothesis(data, theat):
        """
        根据当前参数计算逻辑回归的预测结果。

        参数:
        data: 输入数据。
        theat: 参数向量。

        返回:
        预测结果。
        """
        predictions = sigmoid(np.dot(data, theat))
        return predictions

    @staticmethod
    def gradient_step(data, labels, theta):
        """
        计算逻辑回归模型的梯度。

        参数:
        data: 训练数据。
        labels: 数据标签。
        theta: 参数向量。

        返回:
        当前参数下的梯度向量。
        """
        num_examples = labels.shape[0]
        predictions = LogisticRegression.hypothesis(data, theta)
        label_diff = predictions - labels
        gradients = (1 / num_examples) * np.dot(data.T, label_diff)

        return gradients.T.flatten()

    def predict(self, data):
        """
        使用训练好的逻辑回归模型进行预测。

        参数:
        data: 需要预测的数据。

        返回:
        预测结果。
        """
        num_examples = data.shape[0]
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[
            0]
        prob = LogisticRegression.hypothesis(data_processed, self.theta.T)
        max_prob_index = np.argmax(prob, axis=1)
        class_prediction = np.empty(max_prob_index.shape, dtype=object)
        for index, label in enumerate(self.unique_labels):
            class_prediction[max_prob_index == index] = label
        return class_prediction.reshape((num_examples, 1))
