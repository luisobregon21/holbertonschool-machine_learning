#!/usr/bin/env python3
''' forward propragation '''

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    '''
    performs forward propagation over a convolutional
    layer of a neural network.
    :A_prev: is a numpy.ndarray of shape (m, h_prev,
    w_prev, c_prev) containing the output of the previous layer
        m: is the number of examples
        h_prev: is the height of the previous layer
        w_prev: is the width of the previous layer
        c_prev: is the number of channels in the previous layer
    :W: is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
    kernels for the convolution
        kh: is the filter height
        kw: is the filter width
        c_prev: is the number of channels in the previous layer
        c_new: is the number of channels in the output
    :b: is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    :activation: is an activation function applied to the convolution
    :padding: is a string that is either same or valid, indicating the type
    of padding used
    :stride: is a tuple of (sh, sw) containing the strides for the convolution
        sh: is the stride for the height
        sw: is the stride for the width
    '''

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph = pw = 0
    else:
        ph = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pw = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))

    oh = int(((h_prev + 2 * ph - kh) / sh) + 1)
    ow = int(((w_prev + 2 * pw - kw) / sw) + 1)

    # convolutional dimension
    conv_image = np.zeros((m, oh, ow, c_new))
    padded_images = np.pad(A_prev, pad_width=((0, 0), (ph, ph),
                                              (pw, pw), (0, 0)),
                           mode='constant')

    for y in range(oh):
        for x in range(ow):
            for z in range(c_new):
                image_slice = padded_images[:, (y*sh):(y*sh)+kh,
                                            (x*sw):(x*sw)+kw, :]
                conv_image[:, y, x, z] = np.tensordot(
                    image_slice, W[:, :, :, z],
                    axes=3
                    )
    return activation(conv_image + b)
