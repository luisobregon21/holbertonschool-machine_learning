#!/usr/bin/env python3
'''calculate the derivative of a polynomial'''


def poly_derivative(poly):
    '''
    poly is a list of coefficients representing a polynomial
    the index of the list represents the power of x that the
    coefficient belongs to
    example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
    '''
    if not isinstance(poly, list) or len(poly) < 1:
        return None

    derivative = []

    for power in range(1, len(poly)):
        if type(poly[power]) is not int:
            return None
        derivative.append(poly[power]*power)
    return derivative
