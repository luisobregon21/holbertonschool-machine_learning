#!/usr/bin/env python3
''' train a model with Keras'''

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
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
    :patience: is the patience used for early stopping:
    :learning_rate_decay: is a boolean that indicates whether learning rate
    :alpha: is the initial learning rate
    :decay_rate: is the decay rate
    :save_best: is a boolean indicating whether to save the model after each
    :filepath: is the file path where the model should be saved
    :verbose: is a boolean that determines
    if output should be printed during training
    :shuffle: is a boolean that determines
    whether to shuffle the batches every epoch.
    '''
    early_stopping_callback = []

    def scheduler(epoch):
        '''
        gets the learning rate of each epoch
        :epoch: is the current epoch
        '''
        return alpha / (1 + decay_rate * epoch)

    if validation_data and early_stopping:
        early_stopping_callback.append(
            K.callbacks.EarlyStopping(patience=patience))

    if validation_data and learning_rate_decay:
        learning_rate_decay_callback = K.callbacks.LearningRateScheduler(
            scheduler, verbose=1)
        early_stopping_callback.append(learning_rate_decay_callback)

    if save_best:
        checkpoint = K.callbacks.ModelCheckpoint(filepath, save_best_only=True)
        early_stopping_callback.append(checkpoint)

    return network.fit(data, labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=early_stopping_callback,
                       validation_data=validation_data,
                       verbose=verbose, shuffle=shuffle)
