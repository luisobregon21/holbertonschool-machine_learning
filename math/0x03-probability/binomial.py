#!/usr/bin/env python3
'''
Binomial that represents a binomial distribution
'''


class Binomial():
    '''
    Class to represents a binomial distribution
    considers probability distributions for which
    there are just two possible outcomes with fixed
    probabilities summing to one.
    '''

    def __init__(self, data=None, n=1, p=0.5):
        '''
        :data: is a list of the data to be used to estimate the distribution
        :n: is the number of Bernoulli trials
        :p: is the probability of a “success”
        '''

        self.n = int(n)
        self.p = float(p)

        if data is None:
            if self.n < 1:
                raise ValueError("n must be a positive value")
            elif self.p <= 0 or self.p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)

            variance = 0
            for idx in range(len(data)):
                variance += (data[idx] - mean) ** 2
            variance = variance / len(data)

            self.p = 1 - (variance / mean)
            self.n = int(round(mean / self.p))
            self.p = mean / self.n

    def pmf(self, k):
        '''
        Method Calculates the value of the PMF
        for a given number of “successes”
        :k: is the number of “successes”
        '''

        k = int(k)
        factor_k = 1
        factor_n = 1
        factor_c = 1
        if k > self.n or k < 0:
            return 0

        for num in range(1, k + 1):
            factor_k *= num
        for num in range(1, self.n + 1):
            factor_n *= num
        for num in range(1, (self.n - k) + 1):
            factor_c *= num
        comb = factor_n / (factor_c * factor_k)
        prob = (self.p ** k) * ((1 - self.p) ** (self.n - k))
        pmf = comb * prob
        return pmf

    def cdf(self, k):
        '''
        Method Calculates the value of the CDF
        for a given number of “successes”
        :k: is the number of “successes”
        '''

        k = int(k)
        if k < 0:
            return 0

        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
