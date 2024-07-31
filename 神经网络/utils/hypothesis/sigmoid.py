"""Sigmoid function"""

import numpy as np


def sigmoid(matrix):
    """Applies sigmoid function to NumPy matrix"""
    # e^x 用于二分类  f(x) = \frac{1}{1 + e^{-x}}
    return 1 / (1 + np.exp(-matrix))
