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

    def cost(self, Y, A):
        '''
        Calculates the cost of the model using logistic regression
        :Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        :A: is a numpy.ndarray with shape (1, m) containing
        the activated output of the neuron for each example
        '''
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(A) + (1-Y) * (np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        '''
        Evaluates the neural networks predictions
        :X: is a numpy.ndarray with shape (nx, m) that contains
        the input data
        :Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        '''
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        '''
        Calculates one pass of gradient descent on the neural network.
        :X: is a numpy.ndarray with shape (nx, m) that contains
        the input data
        :Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        :A1: is the output of the hidden layer
        :A2: is the predicted output
        :alpha: is the learning rate
        '''
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = 1/m * np.matmul(dz2, A1.T)
        db2 = 1/m * np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.matmul(self.__W2.T, dz2) * A1 * (1 - A1)
        dw1 = 1/m * np.matmul(dz1, X.T)
        db1 = 1/m * np.sum(dz1, axis=1, keepdims=True)

        self.__W1 = self.__W1 - alpha * dw1
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - alpha * dw2
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        '''
        Trains the neural network.
        :X: is a numpy.ndarray with shape (nx, m) that contains
        the input data
        :Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        :iterations: is the number of iterations to train over
        :alpha: is the learning rate
        '''
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')

        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        for training in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

        return self.evaluate(X, Y)
