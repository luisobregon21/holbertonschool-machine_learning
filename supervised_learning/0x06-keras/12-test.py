#!/usr/bin/env python3
''' test a Neural Network with Keras'''

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    '''
    test a Neural Network.
    :network: is the network model to test
    :data: is the input data to test the model with
    :labels: are the correct one-hot labels of data
    :verbose: is a boolean that determines if output should be printed
    during the testing process
    '''
    return network.evaluate(data, labels, verbose=verbose)
