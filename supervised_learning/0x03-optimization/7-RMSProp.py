#!/usr/bin/env python3
'''update variable using RMSProp optimization algorithm'''


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    '''
    updates a variable using RMSProp optimization algorithm.
    :alpha: learning rate
    :beta2: RMSProp weight
    :epsilon: small number to avoid division by zero
    :var: numpy.ndarray containing the variable to be updated
    :grad: numpy.ndarray containing the gradient of var
    :s: previous second moment of var
    returns the updated variable and the new moment, respectively
    '''
    s = beta2 * s + (1 - beta2) * grad ** 2
    var = var - alpha * grad / (s ** (1 / 2) + epsilon)
    return var, s
