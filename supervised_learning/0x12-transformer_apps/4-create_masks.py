#!/usr/bin/env python3
"""Module create_masks."""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    '''
    creates all masks for training/validation
    :inputs: is a tf.Tensor of shape (batch_size, seq_len_in)
    that contains the input sentence
    :target: is a tf.Tensor of shape (batch_size, seq_len_out)
    that contains the target sentence
    '''
    # Encoder padding mask
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), dtype=tf.float32)
    encoder_mask = tf.expand_dims(encoder_mask, axis=1)
    encoder_mask = tf.expand_dims(encoder_mask, axis=1)

    # Look ahead mask
    look_ahead_mask = tf.cast(tf.math.greater(
        tf.range(tf.shape(target)[1]), tf.range(
            tf.shape(inputs)[1])[:, tf.newaxis]), dtype=tf.float32)
    look_ahead_mask = tf.expand_dims(look_ahead_mask, axis=1)

    # Decoder padding mask
    decoder_mask = tf.cast(tf.math.equal(target, 0), dtype=tf.float32)
    decoder_mask = tf.expand_dims(decoder_mask, axis=1)
    decoder_mask = tf.expand_dims(decoder_mask, axis=1)

    # Combine the masks
    combined_mask = tf.maximum(look_ahead_mask, decoder_mask)

    return encoder_mask, combined_mask, decoder_mask
