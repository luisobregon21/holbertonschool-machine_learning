#!/usr/bin/env python3
"""RNNDecoder class"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Decode for machine translation.
    :tf (tensor): class inherits from
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor.

        :vocab (int): represents the size of the output vocabulary
        :embedding (int): represents the dimensionality of the embedding
                            vector
        :units (int): represents the number of hidden units in the RNN cell
        :batch (int): represents the batch size
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        :x (tensor): contains the previous word in the target sequence as
                    an index of the target vocabulary
        :s_prev (tensor): contains the previous decoder hidden state
        :hidden_states (tensor): contains the outputs of the encoder

        Returns:
            y: contains the output word as a one hot vector in the target
               vocabulary
            s: contains the new decoder hidden state
        """
        batch, units = s_prev.shape
        attention = SelfAttention(units)
        context, weights = attention(s_prev, hidden_states)
        embeddings = self.embedding(x)
        concat_input = tf.concat([tf.expand_dims(context, 1), embeddings],
                                 axis=-1)
        outputs, hidden = self.gru(concat_input)
        outputs = tf.reshape(outputs, (outputs.shape[0], outputs.shape[2]))
        y = self.F(outputs)
        return y, hidden
