#!/usr/bin/env python3
''' Neural NetWork with Keras'''

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    ''''
    builds a neural network with the Keras library.
    :nx: is the number of input features to the network
    :layers: is a list containing the number of nodes in each layer of the network
    :activations: is a list containing the activation functions used for each layer of the network
    :lambtha: is the L2 regularization parameter
    :keep_prob: is the probability that a node will be kept for dropout
    '''
    
    model = K.Sequential()
    L2 = K.regularizers.l2(lambtha)
    for idx in range(len(layers)) :
        if idx == 0:
            model.add(K.layers.Dense(layers[idx], input_shape=(nx,),
                                     activation=activations[idx],
                                     kernel_regularizer=L2,
                                     name='dense'))
        else:
            model.add(K.layers.Dropout(1 - keep_prob))
            model.add(K.layers.Dense(layers[idx], activation=activations[idx],
                                     kernel_regularizer=L2,
                                     name='dense_' + str(layers[idx])))
    return model