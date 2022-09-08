#!/usr/bin/env python3
'''gradient decent with l2 regularization'''

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    '''
    updates the weights and biases of a neural network
    using gradient descent with L2 regularization.
    :Y: one-hot numpy.ndarray of shape (classes, m)
        classes: number of classes
        m: number of data points
    :weights: dictionary of the weights and biases of the neural network
    :cache: dictionary of the outputs of each layer of the neural network
    :alpha: learning rate
    :lambtha: L2 regularization parameter
    :L: number of layers of the network
    '''
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y
    for idx in range(L, 0, -1):
        A = cache['A' + str(idx - 1)]
        W = weights['W' + str(idx)]
        b = weights['b' + str(idx)]
        dw = np.matmul(dz, A.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        dz = np.matmul(W.T, dz) * (1 - A * A)
        weights['W' + str(idx)] = W - alpha * (dw + (lambtha / m) * W)
        weights['b' + str(idx)] = b - alpha * db
