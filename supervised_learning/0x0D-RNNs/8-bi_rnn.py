#!/usr/bin/env python3
"""Module bi_rnn."""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Perform forward propagation for a bidirectional RNN.

    Args:
        bi_cell (object): will be used for the forward propagation
        X (ndarray): data to be used
        h_0 (ndarray): initial hidden state in the forward direction
        h_t (ndarray): initial hidden state in the backward direction
    Returns:
        H: contains all of the concatenated hidden states
        Y: contains all of the outputs
    """
    t, m, i = X.shape
    time_step = range(t)
    _, h = h_0.shape
    H_f = np.zeros((t+1, m, h))
    H_b = np.zeros((t+1, m, h))
    H_f[0] = h_0
    H_b[t] = h_t
    for ts in time_step:
        H_f[ts+1] = bi_cell.forward(H_f[ts], X[ts])
    for ri in range(t-1, -1, -1):
        H_b[ri] = bi_cell.backward(H_b[ri+1], X[ri])
    H = np.concatenate((H_f[1:], H_b[:t]), axis=-1)
    Y = bi_cell.output(H)
    return H, Y
