#!/usr/bin/env python3
"""Module sdp_attetion."""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculate the scaled dot product attention.
    :Q (tensor): contains the query matrix
    :K (tensor): contains the key matrix
    :V (tensor): contains the value matrix
    :mask (tensor, optional): contains the optional mask. Defaults to None.
    Returns:
        output: contains the scaled dot product attention
        weights: contains the attention weights
    """
    q = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_q = q / tf.math.sqrt(dk)
    if mask is not None:
        scaled_q += (mask * -1e9)
    weights = tf.nn.softmax(scaled_q, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights
