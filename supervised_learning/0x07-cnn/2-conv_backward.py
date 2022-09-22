#!/usr/bin/env python3
''' back propragation '''

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    '''
    performs back propagation over a convolutional
    layer of a neural network.
    :dZ: is a numpy.ndarray of shape (m, h_new, w_new, c_new)
    containing the partial derivatives with respect to the
    unactivated output of the convolutional layer
        m: is the number of examples
        h_new: is the height of the output
        w_new: is the width of the output
        c_new: is the number of channels in the output
    :A_prev: is a numpy.ndarray of shape (m, h_prev,
    w_prev, c_prev) containing the output of the previous layer
        h_prev: is the height of the previous layer
        w_prev: is the width of the previous layer
        c_prev: is the number of channels in the previous layer
    :W: is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
    kernels for the convolution
        kh: is the filter height
        kw: is the filter width
    :b: is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    :padding: is a string that is either same or valid, indicating the type
    of padding used
    :stride: is a tuple of (sh, sw) containing the strides for the convolution
        sh: is the stride for the height
        sw: is the stride for the width
    '''

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    m, h_new, w_new, c_new = dZ.shape
    sh, sw = stride

    if padding == 'valid':
        ph = pw = 0
    else:
        ph = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pw = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))

    dA_prev = np.zeros(A_prev.shape)
    # convolutional dimension
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    A_prev_pad = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant')
    padded_images = np.pad(dA_prev, pad_width=((0, 0), (ph, ph), (pw, pw),
                                               (0, 0)), mode='constant')

    for idx in range(m):
        a_prev_pad = A_prev_pad[idx]
        da_prev_pad = padded_images[idx]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw
                    a_slice = a_prev_pad[v_start:v_end, h_start:h_end]
                    da_prev_pad[v_start:v_end, h_start:h_end] +=\
                        W[:, :, :, c] * dZ[idx, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[idx, h, w, c]

        if padding == 'same':
            dA_prev[idx, :, :, :] += da_prev_pad[ph:-ph, pw:-pw, :]
        if padding == 'valid':
            dA_prev[idx, :, :, :] += da_prev_pad

    return dA_prev, dW, db
