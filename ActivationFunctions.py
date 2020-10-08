from numpy import exp as np_exp

def sigmoid(x):
    """
    Compute the sigmoid of x
    Y = 1 / (1 + exp(-x))
    Arguments:
    z -- A scalar or numpy array of any size
    Return:
    s -- sigmoid(x)
    """

    s = 1 / (1 + np_exp(-x))
    
    return s
    
def sigmoid_grad(x):
    """
    Compute the derivative of sigmoid of x
    d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
    Arguments:
    z -- A scalar or numpy array of any size
    Return:
    s -- d_sigmoid(x)/dx
    """

    s = sigmoid(x) * (1 - sigmoid(x))
    
    return s
