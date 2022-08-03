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

    if not any(isinstance(val, int) for val in poly):
        return None
    derivative = []
    for power, coefficient in enumerate(poly):
        if power == 0:
            derivative.append(0)
        if power == 1:
            derivative = []
        derivative.append(power * coefficient)
    return derivative
