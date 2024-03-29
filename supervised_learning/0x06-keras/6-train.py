#!/usr/bin/env python3
''' train a model with Keras'''

import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    '''
    trains a model using mini-batch gradient decent.
    :network: is the model to train
    :data: is a numpy.ndarray of shape (m, nx) containing the input data
    :labels: is a one-hot numpy.ndarray of shape (m, classes) containing
    :batch_size: is the size of the batch used for mini-batch gradient descent
    :epochs: is the number of passes through
    :validation_data: is the data to validate the model with, if not None
    data for mini-batch gradient descent
    :early_stopping: is a boolean that indicates whether early stopping
    :verbose: is a boolean that determines
    :patience: is the patience used for early stopping:
    if output should be printed during training
    :shuffle: is a boolean that determines
    whether to shuffle the batches every epoch.
    '''
    early_stopping_callback = []
    if validation_data and early_stopping:
        early_stopping_callback.append(
            K.callbacks.EarlyStopping(patience=patience))

    return network.fit(data, labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=early_stopping_callback,
                       validation_data=validation_data,
                       verbose=verbose, shuffle=shuffle)
