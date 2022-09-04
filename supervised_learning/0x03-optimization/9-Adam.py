#!/usr/bin/env python3
'''update using Adam optimization algorithm'''


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    '''
    updates a variable in place using the Adam optimization algorithm
    :alpha: learning rate
    :beta1: weight used for the first moment
    :beta2: weight used for the second moment
    :epsilon: small number to avoid division by zero
    :var: numpy.ndarray containing the variable to be updated
    :grad: numpy.ndarray containing the gradient of var
    :v: previous first moment of var
    :s: previous second moment of var
    :t: time step used for bias correction
    return: updated variable, new first moment, new second moment
    '''
    vdw = (beta1 * v) + ((1 - beta1) * grad)
    sdw = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    vdw_corrected = vdw / (1 - (beta1 ** t))
    sdw_corrected = sdw / (1 - (beta2 ** t))
    var = var - (alpha * (vdw_corrected /
                 ((sdw_corrected ** (1 / 2)) + epsilon)))
    return var, vdw, sdw
