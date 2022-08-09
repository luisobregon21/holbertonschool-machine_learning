#!/usr/bin/env python3
'''
Normal that represents a normal distribution
'''


class Normal():
    '''
    Class to represents a normal distribution
    '''

    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        '''
        :data: is a list of the data to be used to estimate the distribution
        :mean: is the mean of the distribution
        :stddev: is the standard deviation of the distribution
        '''

        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            sigma = 0
            for i in range(0, len(data)):
                x = (data[i] - self.mean) ** 2
                sigma += x
            self.stddev = (sigma / len(data)) ** (1 / 2)

    def z_score(self, x):
        '''
        method finds the z score
        :x: is the x-value
        '''

        return (x - self.mean) / self.stddev

    def x_value(self, z):
        '''
        method Calculates the x-value of a given z-score
        :z: is the z-score
        '''
        return self.stddev * z + self.mean

    def pdf(self, x):
        '''
        method Calculates the value of the PDF for a given x-value
        :x: is the x-value
        '''

        p1 = 1 / (self.stddev * ((2 * Normal.pi) ** 0.5))
        p2 = ((x - self.mean) ** 2) / (2 * (self.stddev ** 2))
        return p1 * Normal.e ** (-p2)

    def cdf(self, x):
        '''
        method calculates the value of the CDF for a given x-value
        :x: is the x-value
        '''
        cu = (x - self.mean) / ((2 ** 0.5) * self.stddev)
        errof = (((4 / Normal.pi) ** 0.5) * (cu - (cu ** 3) / 3 +
                                             (cu ** 5) / 10 - (cu ** 7) / 42 +
                                             (cu ** 9) / 216))
        cdf = (1 + errof) / 2
        return cdf
