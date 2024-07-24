"""Sigmoid gradient function"""

from .sigmoid import sigmoid


def sigmoid_gradient(matrix):
    """
    Calculates the gradient of the sigmoid function for a given matrix.

    Parameters:
    matrix: A two-dimensional array (matrix) or tensor, represents the input value of the sigmoid function.

    Returns:
    A two-dimensional array (matrix) or tensor with the same shape as the input, representing the gradient of the sigmoid function at each point in the input matrix.
    """
    """Computes the gradient of the sigmoid function evaluated at z."""
    # Calculate the gradient of the sigmoid function using the formula sigmoid(z) * (1 - sigmoid(z))
    return sigmoid(matrix) * (1 - sigmoid(matrix))
