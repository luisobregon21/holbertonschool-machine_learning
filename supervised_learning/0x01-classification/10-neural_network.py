#!/usr/bin/env python3
''' NeuralNetwork that defines a neural network '''
import numpy as np


class NeuralNetwork():
    '''
    NeuralNetwork class defines a neural network
    with one hidden layer performing binary classification
    '''

    def __init__(self, nx, nodes):
        '''
        class constructor.
        :nx: is the number of input features
        :nodes: is the number of nodes found
        in the hidden layer

        '''

        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        '''
        W1: The weights vector for the hidden layer.
        '''
        return self.__W1

    @property
    def b1(self):
        '''
        b1: The bias for the hidden layer.
        '''
        return self.__b1

    @property
    def A1(self):
        '''
        A1: The activated output for the hidden layer.
        '''
        return self.__A1

    @property
    def W2(self):
        '''
        W2: The weights vector for the output neuron.
        '''
        return self.__W2

    @property
    def b2(self):
        '''
        b2: The bias for the output neuron.
        '''
        return self.__b2

    @property
    def A2(self):
        '''
        A2: The activated output for the output neuron (prediction).
        '''
        return self.__A2

    def forward_prop(self, X):
        '''
        Calculates the forward propagation of the neural network
        :X: is a numpy.ndarray with shape (nx, m) that contains
        the input data
        nx: is the number of input features to the neuron
        m: is the number of examples
        '''
        fwp = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1/(1+np.exp(-fwp))

        self.__A2 = 1 / \
            (1 + np.exp(-(np.matmul(self.__W2, self.__A1) + self.__b2)))
        return self.__A1, self.__A2
