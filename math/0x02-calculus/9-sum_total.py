#!/usr/bin/env python3
''' calculates \\sum_{i=1}^{n} i^2'''


def summation_i_squared(n):
    '''
    calculate sigma sum:
    i = 1, n=n, i^2
    '''

    if not isinstance(n, int) or n < 1:
        return None
    elif n == 1:
        return 1
    return int(n**2) + int(summation_i_squared(n-1))
