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
        self.W = np.random.randn(1, nx)
        # The bias for the neuron.
        self.b = 0
        # The activated output of the neuron (prediction).
        self.A = 0
