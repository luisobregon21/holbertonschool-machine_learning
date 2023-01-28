#!/usr/bin/env python3
"""Module MultiHeadAttention class."""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Perform multi head attention.
    Args:
        tf (tensor): class inherit from
    """

    def __init__(self, dm, h):
        """
        Class constructor.
        Args:
            dm (int): represents the dimensionality of the model
            h (int): represents the number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """
        Call method.
        Args:
            Q (tensor): contains the input to generate the query matrix
            K (tensor): contains the input to generate the key matrix
            V (tensor): contains the input to generate the value matrix
            mask (tensor): is always None
        Returns:
            output: contains the scaled dot product attention
            weights: contains the attention weights
        """
        batch_size = tf.shape(Q)[0]
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = sdp_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1,
                                                         self.dm))
        output = self.linear(concat_attention)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        """Split heads method."""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
