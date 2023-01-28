#!/usr/bin/env python3
"""Class SelfAttention"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Calculate the attention for machine translation.

    :tf (tensor): class inhertis from
    """

    def __init__(self, units):
        """
        :units (int): represents the number of hidden units in the
        alignment model
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        :s_prev (tensor): contains the previous decoder hidden state
        :hidden_states (tensor): contains the outputs of the encoder

        Returns:
            context: contains the context vector for the decoder
            weights: contains the attention weights
        """
        new_s_prev = tf.expand_dims(s_prev, axis=1)
        score = self.V(tf.nn.tanh(self.W(new_s_prev) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
