#!/usr/bin/env python3
"""Class RNNEncoder"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    Encode for machine translation.
    :tf (tensor): class inherits from
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        :vocab (int): represents the size of the input vocabulary
        :embedding (int): represents the dimensionality of the embedding
                            vector
        :units (int): represents the number of hidden units in the RNN cell
        :batch (int): represents the batch size
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
        Initialize the hidden states for the RNN cell to a tensor of zeros.
        Returns:
            tensor containing the initialized hidden states
        """
        initiliazer = tf.keras.initializers.Zeros()
        hiddenQ = initiliazer(shape=(self.batch, self.units))
        return hiddenQ

    def call(self, x, initial):
        """
        Call methods.
        :x (tensor): contains the input to the encoder layer
        :initial (tensor): contains the initial hidden state
        Returns:
            outputs: contains the outputs of the encoder
            hidden: contains the last hidden state of the encoder
        """
        embedding = self.embedding(x)
        outputs, hidden = self.gru(embedding, initial_state=initial)
        return outputs, hidden
