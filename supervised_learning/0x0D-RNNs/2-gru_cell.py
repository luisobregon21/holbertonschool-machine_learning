#!/usr/bin/env python3
"""Class GRUCell."""
import numpy as np


class GRUCell:
    """Represent a gated recurrent unit."""

    def __init__(self, i, h, o):
        """
        Class constructor.

        Args:
            i (int): dimensionality of the data
            h (int): dimensionality of the hidden state
            o (int): dimensionality of the outputs
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev (ndarray): contains the previous hidden state
            x_t (ndarray): contains the data input for the cell
        Returns:
            h_next: next hidden state
            y: output of the cell
        """
        h_x = np.concatenate((h_prev.T, x_t.T), axis=0)
        zt = self.sigmoid((h_x.T @ self.Wz) + self.bz)
        rt = self.sigmoid((h_x.T @ self.Wr) + self.br)
        h_x = np.concatenate(((rt * h_prev).T, x_t.T), axis=0)
        ht_c = np.tanh((h_x.T @ self.Wh) + self.bh)
        h_next = (1 - zt) * h_prev + zt * ht_c
        y = self.softmax((h_next @ self.Wy) + self.by)
        return h_next, y

    def sigmoid(self, x):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """Softmax function."""
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
