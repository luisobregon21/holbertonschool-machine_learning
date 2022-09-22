#!/usr/bin/env python3
'''modified LeNet-5 with Tensorflow'''

import tensorflow.compat.v1 as tf


def lenet5(x, y):
    '''
    builds a modified version of the LeNet-5
    architecture usinhg tensorflow.
    :x: is a tf.placeholder of shape (m, 28, 28, 1)
    containing the input images for the network
        m: is the number of images
    :y: is a tf.placeholder of shape (m, 10) containing
    the one-hot labels for the network
    '''
    init = tf.keras.initializers.VarianceScaling(scale=2.0)
    # relu activation
    activation = tf.nn.relu
    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = tf.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                             activation=activation, kernel_initializer=init)(x)
    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool1 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)
    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                             activation=activation,
                             kernel_initializer=init)(pool1)
    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)
    flatten = tf.layers.Flatten()(pool2)
    # Fully connected layer with 120 nodes
    FC1 = tf.layers.Dense(units=120, activation=activation,
                          kernel_initializer=init)(flatten)
    # Fully connected layer with 84 nodes
    FC2 = tf.layers.Dense(units=84, activation=activation,
                          kernel_initializer=init)(FC1)
    # Fully connected softmax output layer with 10 nodes
    FC3 = tf.layers.Dense(units=10, kernel_initializer=init)(FC2)
    y_pred = FC3
    #  tensor for the loss of the netowrk
    loss = tf.losses.softmax_cross_entropy(y, FC3)
    # training operation that utilizes Adam optimization
    train = tf.train.AdamOptimizer().minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    # tensor for the accuracy of the network
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # a tensor for the softmax activated output
    y_pred = tf.nn.softmax(y_pred)
    return y_pred, train, loss, accuracy
