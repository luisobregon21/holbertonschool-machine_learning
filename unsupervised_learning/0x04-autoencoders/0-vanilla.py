#!/usr/bin/env python3
"""Module autoencoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Create an autoencoder.

    :input_dims (int): contains the dimensions of the model input
    :hidden_layers (list): contains the number of nodes for each hidden
    layer in the encoder, for the decoder they
    should be reversed
    :latent_dims (int): contains the dimensions of the latent space
    representation
    Returns:
        encoder: encoder model
        decoder: decoder model
        auto: full autoencoder model
    """
    X_input_encoded = keras.Input(shape=(input_dims,))
    hidden_ly = keras.layers.Dense(units=hidden_layers[0], activation='relu')
    Y_prev = hidden_ly(X_input_encoded)
    for i in range(1, len(hidden_layers)):
        hidden_ly = keras.layers.Dense(units=hidden_layers[i],
                                       activation='relu')
        Y_prev = hidden_ly(Y_prev)
    latent_ly = keras.layers.Dense(units=latent_dims, activation='relu')
    bottleneck = latent_ly(Y_prev)
    encoder = keras.Model(X_input_encoded, bottleneck)
    X_input_decoded = keras.Input(shape=(latent_dims,))
    hidden_ly_decoded = keras.layers.Dense(units=hidden_layers[-1],
                                           activation='relu')
    Y_prev = hidden_ly_decoded(X_input_decoded)
    for j in range(len(hidden_layers) - 2, -1, -1):
        hidden_ly_decoded = keras.layers.Dense(units=hidden_layers[j],
                                               activation='relu')
        Y_prev = hidden_ly_decoded(Y_prev)
    last_layer = keras.layers.Dense(units=input_dims, activation='sigmoid')
    output = last_layer(Y_prev)
    decoder = keras.Model(X_input_decoded, output)
    X_input = keras.Input(shape=(input_dims,))
    encoder_o = encoder(X_input)
    decoder_o = decoder(encoder_o)
    auto = keras.Model(X_input, decoder_o)
    auto.compile(loss='binary_crossentropy', optimizer='adam')
    return encoder, decoder, auto
