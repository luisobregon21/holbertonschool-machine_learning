#!/usr/bin/env python3
""" Transformer class """
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    Create a transformer network.
    :tf (tensor): class inherits from
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        :N (int): number of blocks in the encoder and decoder
        :dm (int): dimensionality of the model
        :h (int): number of heads
        :hidden (int): number of hidden units in the fully connected layers
        :input_vocab (int): size of the input vocabulary
        :target_vocab (int): size of the tarhet vocabulary
        :max_seq_input (int): maximum sequence length possible for the input
        :max_seq_target (int): maximum sequence length possible for the
                                target
        :drop_rate (float, optional): dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input,
                               drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_target,
                               drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
        Call method.

        :inputs (tensor): contains the inputs
        :target (tensor): contains the target
        :training (boolean): determines if the model is training
        :encoder_mask (tensor): mask to be applied to the encoder
        :look_ahead_mask (tensor): mask to be applied to the decoder
        :decoder_mask (tensor): mask to be applied to the decoder
        Returns:
            tensor containing the transformer output
        """
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)
        final_output = self.linear(dec_output)
        return final_output
