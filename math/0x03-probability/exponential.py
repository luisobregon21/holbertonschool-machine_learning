#!/usr/bin/env python3
'''
module has Exponential that represents an exponential distribution
'''


class Exponential():
    ''' 
    class to represent Exponential distribution.
    models the time elapsed between events.
    '''

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        '''
        class constructor. Initialize method
        data.

        :param data: list of the data to be used to estimate the distribution
        :param lambtha: the expected number of occurences in a given time frame
        '''

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = (sum(data) / len(data))

    def pdf(self, x):
        '''
        Calculates the value of the PDF for a given time period
        :x: is the time period
        '''

        if x < 0:
            return 0

        pdf = self.lambtha * (Exponential.e ** ((-self.lambtha) * x))
        return pdf

    def cdf(self, x):
        '''
        Calculates the value of the CDF for a given time period
        :x: is the time period
        '''

        if x < 0:
            return 0
        cdf = 1 - (Exponential.e ** ((-self.lambtha) * x))
        return cdf
