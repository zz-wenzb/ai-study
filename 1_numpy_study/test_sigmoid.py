import numpy as np


def sigmoid(x):
    """
    Applies the sigmoid function element-wise to a NumPy array.

    Parameters:
    x : numpy.ndarray
        Input array or scalar.

    Returns:
    numpy.ndarray
        The result of applying the sigmoid function to each element of x.
    """
    """
    Sigmoid 函数是一种常用的激活函数，在神经网络和机器学习中被广泛应用，尤其是在二分类问题中。
    Sigmoid 函数的数学形式为 ( f(x) = \frac{1}{1 + e^{-x}} )，
    其图形是一个 S 形曲线，能够将任意实数映射到 (0, 1) 区间内，这非常适合用作概率估计。
    """
    return 1 / (1 + np.exp(-x))


# 创建一个 NumPy 数组
data = np.array([[-1, 0, 1], [2, 3, 4]])

# 应用 Sigmoid 函数
result = sigmoid(data)
print(result)
