#!/usr/bin/env python3
'''gradient descent with momentum optimization algorithm'''


def update_variables_momentum(alpha, beta1, var, grad, v):
    '''
    updates a variable using the gradient descent with momentum
    optimization algorithm.
    :alpha: learning rate
    :beta1: momentum weight
    :var: numpy.ndarray - contains the variable to be updated
    :grad: numpy.ndarray - contains the gradient of var
    :v: previous first moment of var
    Returns: the updated variable and the new moment, respectively
    '''
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
