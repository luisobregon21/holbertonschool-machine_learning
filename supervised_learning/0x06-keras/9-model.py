#!/usr/bin/env python3
''' load and save model with Keras'''

import tensorflow.keras as K


def save_model(network, filename):
    '''
    saves an entire model.
    :network: is the model to save
    :filename: is the path of the file that the model should be saved to
    '''
    network.save(filename)
    return None


def load_model(filename):
    '''
    loads an entire model.
    :filename: is the path of the file that the model should be loaded from
    '''
    return K.models.load_model(filename)
