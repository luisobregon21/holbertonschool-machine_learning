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

    def forward_prop(self, X):
        '''
        Calculates the forward propagation of the neuron
        :X: is a numpy.ndarray with shape (nx, m) that contains the input data
        (nx is the # of input features to the neuron, m is the # of examples)
        '''
        fwp = np.matmul(self.__W, X) + self.__b
        # sigmoid activation
        self.__A = 1/(1+np.exp(-fwp))
        return self.__A

    def cost(self, Y, A):
        '''
        Calculates the cost of the model using logistic regression
        :Y: is a numpy.ndarray with shape (1, m) that contains the
        correct labels for the input data

        :A: is a numpy.ndarray with shape (1, m) containing the
        activated output of the neuron for each example
        '''

        # to avoid division by zero error: 1.0000001 - A
        m = Y.shape[1]
        total_cost = -(1 / m) * np.sum(np.multiply(Y, np.log(A)) +
                                       np.multiply(1 - Y,
                                                   np.log(1.0000001 - A)))
        return total_cost

    def evaluate(self, X, Y):
        '''
        Evaluates the neurons predictions
        :X: is a numpy.ndarray with shape (nx, m) that contains
        the input data.
        nx is the number of input features to the neuron
        m is the number of examples

        :Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        '''
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        # The label values should be 1 if the output
        # of the network is >= 0.5 and 0 otherwise
        prediction = np.where(self.__A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''
        Calculates one pass of gradient descent on the neuron

        :X: is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples

        :Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        :A: is a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example

        alpha is the learning rate
        '''
        m = Y.shape[1]
        derivative_z = A - Y
        print()
        dW = np.matmul(X, derivative_z) / m
        db = np.sum(derivative_z) / m
        self.__W = self.__W - (dW * alpha)
        self.__b = self.__b - (alpha * db)
