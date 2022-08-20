#!/usr/bin/env python3
'''Deep Neural Network'''

import numpy as np


class DeepNeuralNetwork():
    '''
    class defines deep neural network
    performing binary classification
    '''

    def __init__(self, nx, layers):
        '''
        class constructor
        :nx: number of input features
        :layers:  list representing the number
        of nodes in each layer of the network

        The first value in layers represents
        the number of nodes in the first layer.
        '''
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for layer in range(self.__L):
            if type(layers[layer]) is not int or layers[layer] < 1:
                raise TypeError('layers must be a list of positive integers')

            self.__weights['b' + str(layer + 1)] = np.zeros(
                (layers[layer], 1))
            if layer == 0:
                He_et_al = np.random.randn(layers[layer], nx) * np.sqrt(2 / nx)
                self.__weights['W' + str(layer + 1)] = He_et_al
            else:
                He_et_al = np.random.randn(
                    layers[layer], layers[layer - 1]) * np.sqrt(
                        2 / layers[layer - 1])
                self.__weights['W' + str(layer + 1)] = He_et_al

    @property
    def L(self):
        '''The number of layers in the neural network.'''
        return self.__L

    @property
    def cache(self):
        '''A dictionary to hold all intermediary values of the network.'''
        return self.__cache

    @property
    def weights(self):
        '''A dictionary to hold all weights and biased of the network.'''
        return self.__weights
