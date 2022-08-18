#!/usr/bin/env python3
''' holds Neuron class that defines a single neuron '''
import numpy as np


class Neuron():
    '''
    class Neuron that defines a single neuron
    performing binary classification
    '''

    def __init__(self, nx):
        '''
        class constructor
        :nx: number of input features to the neuron
        '''
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # The weights vector for the neuron.
        self.__W = np.random.randn(nx).reshape(1, nx)
        # The bias for the neuron.
        self.__b = 0
        # The activated output of the neuron (prediction).
        self.__A = 0

    @property
    def W(self):
        ''' private instance weight '''
        return self.__W

    @property
    def b(self):
        ''' private instance bias '''
        return self.__b

    @property
    def A(self):
        ''' Returns: private instance output '''
        return self.__A
