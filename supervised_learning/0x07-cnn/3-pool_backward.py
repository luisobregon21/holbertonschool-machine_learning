#!/usr/bin/env python3
''' back propragation '''

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    '''
    performs back propagation over a pooling
    layer of a neural network.
    :dA: is a numpy.ndarray of shape (m, h_new, w_new, c_new)
    containing the partial derivatives with respect to the
    output of the pooling layer
        m: is the number of examples
        h_new: is the height of the output
        w_new: is the width of the output
        c: is the number of channels
    :A_prev: is a numpy.ndarray of shape (m, h_prev,
    w_prev, c_prev) containing the output of the previous layer
        m: is the number of examples
        h_prev: is the height of the previous layer
        w_prev: is the width of the previous layer
        c_prev: is the number of channels in the previous layer
    :kernel_shape: is a tuple of (kh, kw) containing the size of
    the kernel for the pooling
        kh: is the kernel height
        kw: is the kernel width
    :stride: is a tuple of (sh, sw) containing the strides for the convolution
        sh: is the stride for the height
        sw: is the stride for the width
    :mode: is a string containing either max or avg, indicating whether
    to perform maximum or average pooling, respectively
    '''

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    m, h_new, w_new, c_new = dA.shape
    dA_prev = np.zeros(A_prev.shape)

    for idx in range(m):
        a_prev = A_prev[idx]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw
                    if mode == 'max':
                        a_slice = a_prev[v_start:v_end, h_start:h_end, c]
                        mask = (a_slice == np.max(a_slice))
                        dA_prev[idx, v_start:v_end, h_start:h_end, c] +=\
                            np.multiply(mask, dA[idx, h, w, c])
                    else:
                        da = dA[idx, h, w, c]
                        shape = kernel_shape
                        avg = da / (kh * kw)
                        Z = np.ones(shape) * avg
                        dA_prev[idx, v_start:v_end, h_start:h_end, c] += Z
    return dA_prev
