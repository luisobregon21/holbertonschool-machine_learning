#!/usr/bin/env python3
'''0. L2 Regularization Cost'''

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    '''
    calculates the cost of a neural network with L2 regularization
    :cost: is the cost of the network without L2 regularization
    :lambtha: is the regularization parameter
    :weights: is a dictionary of the weights and biases (numpy.ndarrays)
    :L: is the number of layers in the neural network
    :m: is the number of data points used
    '''
    l2 = 0
    for idx in range(1, L + 1):
        l2 += np.linalg.norm(weights['W' + str(idx)])
    return cost + (l2 * lambtha / (2 * m))
