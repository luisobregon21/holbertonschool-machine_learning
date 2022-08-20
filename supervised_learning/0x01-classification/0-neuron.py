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

        W: The weights vector for the neuron.
        b:`The bias for the neuron.
        A: The activated output of the neuron (prediction).
        '''
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
