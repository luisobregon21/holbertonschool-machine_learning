#!/usr/bin/env python3
''' calculates \\sum_{i=1}^{n} i^2'''


def summation_i_squared(n):
    '''
    calculate sigma sum:
    i = 1, n=n, i^2
    '''

    if n == 1:
        return 1
    elif n > 0:
        return int((n**2) + (summation_i_squared(n-1)))
    else:
        return None
