#!/usr/bin/env python3
''' forward propragation '''

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    '''
    performs forward propagation over a pooling
    layer of a neural network.
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

    oh = int((h_prev - kh) / sh) + 1
    ow = int((w_prev - kw) / sw) + 1

    # convolutional dimension
    conv_image = np.zeros((m, oh, ow, c_prev))

    for y in range(oh):
        for x in range(ow):
            if mode == 'max':
                conv_image[:, x, y] = (np.max(A_prev[:,
                                                     x*sh:((x*sh)+kh),
                                                     y*sw:((y*sw)+kw)],
                                              axis=(1, 2)))
            else:
                conv_image[:, x, y] = (np.mean(A_prev[:,
                                                      x*sh:((x*sh)+kh),
                                                      y*sw:((y*sw)+kw)],
                                               axis=(1, 2)))
    return conv_image
