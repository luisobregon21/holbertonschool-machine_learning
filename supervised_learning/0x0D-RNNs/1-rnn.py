#!/usr/bin/env python3
"""Module rnn."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Perform forward propagation for a simple RNN.

    Args:
        rnn_cell (object): used for the forward propagation
        X (ndarray): data to be used
        h_0 (ndarray): initial hidden state
    Returns:
        H: contains all of the hidden states
        Y: contains all of the outputs
    """
    Y = []
    t, m, i = X.shape
    time_step = range(t)
    _, h = h_0.shape
    H = np.zeros((t+1, m, h))
    H[0, :, :] = h_0
    for ts in time_step:
        h_next, y_pred = rnn_cell.forward(H[ts], X[ts])
        H[ts+1, :, :] = h_next
        Y.append(y_pred)
    Y = np.array(Y)
    return H, Y
