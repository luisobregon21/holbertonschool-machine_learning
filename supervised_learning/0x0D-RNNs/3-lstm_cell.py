#!/usr/bin/env python3
"""Class LSTMCell."""
import numpy as np


class LSTMCell:
    """Represent an LSTM unit."""

    def __init__(self, i, h, o):
        """
        Class constructor.

        Args:
            i (int): dimensionality of the data
            h (int): dimensionality of the hidden state
            o (int): dimensionality of the outputs
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev (ndarray): contains the previous hidden state
            c_prev (ndarray): contains the previous cell state
            x_t (ndarray): contains the data input for the cell
        Returns:
            h_next: next hidden state
            c_next: next cell state
            y: output of the cell
        """
        h_x = np.concatenate((h_prev.T, x_t.T), axis=0)
        ft = self.sigmoid((h_x.T @ self.Wf) + self.bf)
        it = self.sigmoid((h_x.T @ self.Wu) + self.bu)
        cct = np.tanh((h_x.T @ self.Wc) + self.bc)
        c_next = ft * c_prev + it * cct
        ot = self.sigmoid((h_x.T @ self.Wo) + self.bo)
        h_next = ot * np.tanh(c_next)
        y = self.softmax((h_next @ self.Wy) + self.by)
        return h_next, c_next, y

    def sigmoid(self, x):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """Softmax function."""
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
