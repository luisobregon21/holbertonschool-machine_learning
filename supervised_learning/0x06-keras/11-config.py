#!/usr/bin/env python3
''' load and save a model configuration with Keras'''

import tensorflow.keras as K


def save_config(network, filename):
    '''
    saves a models configuration.
    :network: is the model whose configuration should be saved
    :filename: is the path of the file that the model should be saved to
    '''
    with open(filename, 'w') as f:
        f.write(network.to_json())
    f.close()
    return None


def load_config(filename):
    '''
    loads a model with specific configuration.
    :network: is the model to which the configuration should be loaded
    :filename: is the path of the file that the model should be loaded from
    '''
    with open(filename, 'r') as f:
        model = K.models.model_from_json(f.read())
    f.close()
    return model
