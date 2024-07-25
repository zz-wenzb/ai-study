"""Sigmoid function"""

import numpy as np


def sigmoid(matrix):
    """
    Calculates the sigmoid function for a given NumPy matrix.

    The sigmoid function is a type of mathematical function that maps values from a given domain to values in a specified range,
    typically between 0 and 1. It is often used in machine learning models as an activation function to introduce non-linearity.

    Parameters:
    matrix: A NumPy array or matrix of arbitrary shape. This function can handle multi-dimensional arrays,
            allowing the sigmoid function to be applied element-wise to the input array.

    Returns:
    A NumPy array or matrix of the same shape as the input, with the sigmoid function applied to each element.
    """
    """Applies sigmoid function to NumPy matrix"""
    # Apply the sigmoid function to each element in the matrix
    return 1 / (1 + np.exp(-matrix))
