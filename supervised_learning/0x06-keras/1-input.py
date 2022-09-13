#!/usr/bin/env python3
''' Neural NetWork with Keras'''

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    ''''
    builds a neural network with the Keras library not using
    the sequential class.
    :nx: is the number of input features to the network
    :layers: is a list containing the number of nodes
    in each layer of the network
    :activations: is a list containing the activation
    functions used for each layer of the network
    :lambtha: is the L2 regularization parameter
    :keep_prob: is the probability that a node will be kept for dropout
    '''

    inputs = K.Input(shape=(nx,))
    L2 = K.regularizers.l2(lambtha)
    for idx in range(len(layers)):
        if idx == 0:
            output = K.layers.Dense(layers[idx],
                                    activation=activations[idx],
                                    kernel_regularizer=L2)(inputs)
        else:
            dropout = K.layers.Dropout(1 - keep_prob)(output)
            output = K.layers.Dense(layers[idx], activation=activations[idx],
                                    kernel_regularizer=L2)(dropout)
    return K.models.Model(inputs=inputs, outputs=output)