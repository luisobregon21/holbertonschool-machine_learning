#!/usr/bin/env python3
''' optimization with Keras'''

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    '''
    sets up Adam optimization for a keras model
    with categorical crossentropy loss and accuracy metrics.
    :network: model to optimize
    :alpha: learning rate
    :beta1: first Adam optimization parameter
    :beta2: second Adam optimization parameter
    :returns: None
    '''
    network.compile(optimizer=K.optimizers.Adam(lr=alpha,
                                                beta_1=beta1,
                                                beta_2=beta2),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return None
