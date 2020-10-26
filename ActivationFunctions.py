from numpy import exp as np_exp


def sigmoid(x, alpha=1):
    """
    Compute the sigmoid of x
    Y = 1 / (1 + exp(-alpha*x))

    Arguments:
    x -- A scalar or numpy array of any size
    alpha -- A scalar of size (1,1)

    Return:
    y -- sigmoid(x)
    """

    y = 1 / (1 + np_exp(-alpha*x))
    
    return y


def sigmoid_grad(x, alpha=1):
    """
    Compute the derivative of sigmoid of x
    d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
    
    Arguments:
    x -- A scalar or numpy array of any size
    alpha -- A scalar of size (1,1)

    Return:
    y -- d_sigmoid(x)/dx
    """

    y = sigmoid(x, alpha) * (1 - sigmoid(x, alpha))
    
    return y


def relu(x):
    """
    Compute the Rectified Linear Unit of x
    y=x if x > 0, else x = 0
    
    Arguments:
    x -- A scalar or numpy array of any size
    Return:
    y -- relu(x)
    """

    y = x * (x > 0)

    return y


def relu_grad(x):
    """
    Compute the Derivative of Rectified Linear Unit of x
    y=1.0 if x > 0, else x = 0
    
    Arguments:
    x -- A scalar or numpy array of any size
    Return:
    y -- d_relu(x)/dx
    """

    y = 1. * (x > 0)

    return y


def tanh(x):
    """
    Compute the Tangent hyperbolic of x
    
    Arguments:
    x -- A scalar or numpy array of any size
    Return:
    y -- tanh(x)
    """

    y = (np_exp(x) - np_exp(-x)) / (np_exp(x) + np_exp(-x))

    return y


def tanh_grad(x):
    """
    Compute the Derivative of Tangent hyperbolic of x
    y = 1 - tanh(x)^2
    
    Arguments:
    x -- A scalar or numpy array of any size
    Return:
    y -- d_tanh(x)/dx
    """

    y = 1 - tanh(x)**2

    return y
