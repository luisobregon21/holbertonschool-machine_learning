#!/usr/bin/env python3
''' calculates \\sum_{i=1}^{n} i^2'''


def summation_i_squared(n):
    '''
    calculate sigma sum:
    i = 1, n=n, i^2
    '''
    if n > 0:
        return int(n*(n+1)*((2*n)+1)/6)
    return None
