#!/usr/bin/env python3
'''
module has Poisson that represents a poisson distribution
'''


class Poisson():
    '''
    class to represent poisson distribution.
    Calculates probabilities of various number
    of "success" based on the mean nmumber of success.

    success means that the outcome in question occurs.
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

    def pmf(self, k):
        '''
        Calculates the value of the PMF for a given number of “successes”
        :k: number of “successes”
        '''
        k = int(k)
        factorial = 1
        if k < 0:
            return 0

        for num in range(1, k+1):
            factorial *= num

        pmf = Poisson.e ** -self.lambtha * self.lambtha ** k / factorial
        return pmf

    def cdf(self, k):
        '''
        Calculates the value of the CDF for a given number of “successes”
        :k: is the number of “successes”
        '''
        k = int(k)
        if k < 0:
            return 0

        cdf = 0
        for num in range(k+1):
            cdf = cdf + self.pmf(num)

        return cdf
