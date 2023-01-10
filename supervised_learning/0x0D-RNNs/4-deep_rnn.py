#!/usr/bin/env python3
"""Module deep_rnn."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Perform forward propagation for a deep RNN.

    Args:
        rnn_cells (list): contains RNNCell instances that will be used for the
                          forward propagation
        X (ndarray): data to be used
        h_0 (ndarray): initial hidden state
    Returns:
        H: contains all of the hidden states
        Y: contains all of the outputs
    """
    Y = []
    t, m, i = X.shape
    _, _, h = h_0.shape
    time_step = range(t)
    layers = len(rnn_cells)
    H = np.zeros((t+1, layers, m, h))
    H[0, :, :, :] = h_0
    for ts in time_step:
        for ly in range(layers):
            if ly == 0:
                h_next, y_pred = rnn_cells[ly].forward(H[ts, ly], X[ts])
            else:
                h_next, y_pred = rnn_cells[ly].forward(H[ts, ly], h_next)
            H[ts+1, ly, :, :] = h_next
        Y.append(y_pred)
    Y = np.array(Y)
    return H, Y
