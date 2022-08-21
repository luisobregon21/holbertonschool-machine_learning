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

    def forward_prop(self, X):
        '''
        Calculates the forward propagation of the neural network
        :X: is a numpy.ndarray with shape (nx, m) that contains
        the input data
        nx: is the number of input features to the neuron
        m: is the number of examples
        '''
        self.__cache["A0"] = X
        for layer in range(self.__L):
            weights = self.__weights
            cache = self.__cache
            fwp_A = np.matmul(
                weights["W" + str(layer + 1)], cache["A" + str(layer)])

            fwp = fwp_A + weights["b" + str(layer + 1)]
            cache["A" + str(layer + 1)] = 1 / (1 + np.exp(-fwp))

        return cache["A" + str(self.__L)], cache

    def cost(self, Y, A):
        '''
        Calculates the cost of the model using logistic regression
        :Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        :A: is a numpy.ndarray with shape (1, m) containing
        the activated output of the neuron for each example
        '''
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        '''
        Evaluates the Neural Network's prediction
        :X: is a numpy.ndarray with shape (nx, m) that contains
        the input data
        :Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        '''
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
