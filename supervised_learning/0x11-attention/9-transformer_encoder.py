#!/usr/bin/env python3
""" Encoder class """
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Create the encoder for a transformer.
    :tf (tensor): class inhertis from
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor.

        :N (int): number of blocks in the encoder
        :dm (int): dimensionality of the model
        :h (int): number of heads
        :hidden (int): number of hidden units in the fully connected layer
        :input_vocab (int): size of the input vocabulary
        :max_seq_len (int): maximum sequence length possible
        :drop_rate (float, optional): dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
                       ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Call method.

        :x (tensor): contains the input to the encoder
        :training (boolean): determines if the model is training
        :mask (tensor): mask to be applied for multi head attention
        Returns:
            tensor containing the encoder output
        """
        seq_len = x.shape[1]
        embedding = self.embedding(x)
        embedding *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        embedding += self.positional_encoding[:seq_len]
        encoder_out = self.dropout(embedding, training=training)
        for i in range(self.N):
            encoder_out = self.blocks[i](encoder_out, training, mask)
        return encoder_out
