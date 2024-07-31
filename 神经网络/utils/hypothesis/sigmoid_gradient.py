"""Sigmoid gradient function"""

from .sigmoid import sigmoid


def sigmoid_gradient(matrix):
    """Computes the gradient of the sigmoid function evaluated at z."""
    """
    f(x) = \sigma(x) = \frac{1}{1 + e^{-x}}
     ( f'(x) ) 可以表示为 ( f(x)(1 - f(x)) )。
    """
    return sigmoid(matrix) * (1 - sigmoid(matrix))
