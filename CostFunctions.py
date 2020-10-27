from numpy import mean as np_mean
from numpy import log as np_log
from numpy import divide as np_divide

EPSILON = 1e-10


def mse(actual_val, estimated_val):
    """
    Compute Mean Square Error

    Y = sum( (actual_val - estimated_val)^2 ) / n

    Arguments:
    actual_val -- A scalar or numpy array of size (Nb data points, Nb features)
    estimated_val -- A scalar or numpy array of size (Nb data points, Nb features)

    Return:
    error -- A scalar or numpy array of size (1, Nb features)
    """

    y = np_mean((estimated_val - actual_val)**2)

    return y


def mse_grad(actual_val, estimated_val):
    """
    Compute Mean Square Error Gradient (derivative)

    Y = sum( 2 * (actual_val - estimated_val) ) / n

    Arguments:
    actual_val -- A scalar or numpy array of size (Nb data points, Nb features)
    estimated_val -- A scalar or numpy array of size (Nb data points, Nb features)

    Return:
    error -- A scalar or numpy array of size (1, Nb features)
    """

    error = np_mean(2 * (estimated_val - actual_val), axis=0, keepdims=True)

    return error


def binary_cross_entropy(actual_val, estimated_val):
    """
    Compute Cross Entropy Error for 2 classes

    Y = - sum( actual_val *log(estimated_val) + (1 - actual_val) * log(1 - estimated_val ) / n

    Arguments:
    actual_val -- A scalar or numpy array of size (Nb data points, Nb features)
    estimated_val -- A scalar or numpy array of size (Nb data points, Nb features)

    Return:
    error -- A scalar or numpy array of size (1, Nb features)
    """

    # Fix issue with log computation. Error would become undefined
    estimated_val[estimated_val == 1.] = 1 - EPSILON
    estimated_val[estimated_val == 0.] = EPSILON

    error = - np_mean(actual_val * np_log(estimated_val) + (1 - actual_val) * np_log(1 - estimated_val))

    return error


def binary_cross_entropy_grad(actual_val, estimated_val):
    """
    Compute Cross Entropy Error Gradient (derivative) for 2 classes

    Y = sum(actual_val / estimated val - (1 - actual_val) / (1 - estimated_val)) / n

    Arguments:
    actual_val -- A scalar or numpy array of size (Nb data points, Nb features)
    estimated_val -- A scalar or numpy array of size (Nb data points, Nb features)

    Return:
    error -- A scalar or numpy array of size (1, Nb features)
    """

    error = - np_mean(np_divide(actual_val, estimated_val) - np_divide(1 - actual_val, 1 - estimated_val), axis=0, keepdims=True)

    return error
