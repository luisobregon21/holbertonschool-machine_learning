#!/usr/bin/env python3
'''modified LeNet-5 with Keras'''

import tensorflow.keras as K


def lenet5(X):
    '''
    builds a modified version of the LeNet-5
    architecture usinhg Keras.
    :X: is a K.Input of shape (m, 28, 28, 1) containing
    the input images for the network
        m: is the number of images
    '''
    init = K.initializers.he_normal()
    activation = 'relu'
    conv1 = K.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                            activation=activation, kernel_initializer=init)(X)
    pool1 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)
    conv2 = K.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                            activation=activation,
                            kernel_initializer=init)(pool1)
    pool2 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)
    flatten = K.layers.Flatten()(pool2)
    FC1 = K.layers.Dense(units=120, activation=activation,
                         kernel_initializer=init)(flatten)
    FC2 = K.layers.Dense(units=84, activation=activation,
                         kernel_initializer=init)(FC1)
    FC3 = K.layers.Dense(units=10, kernel_initializer=init,
                         activation='softmax')(FC2)
    model = K.models.Model(X, FC3)
    adam = K.optimizers.Adam()
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model