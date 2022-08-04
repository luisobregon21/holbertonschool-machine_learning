#!/usr/bin/env python3
''' calculate the integral of a polynomial '''


def poly_integral(poly, C=0):
    '''
    poly is a list of coefficients representing a polynomial
    C is an integer representing the integration constant
    example:
     if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
    '''

    if poly is None or poly == [] or type(poly) != list:
        return None

    if not isinstance(C, (int, float)):
        return None

    if not any(isinstance(val, (int, float)) for val in poly):
        return None

    # C is an integer representing the integration constant
    if isinstance(C, float) and C.is_integer:
        C = int(C)

    integral = [C]

    for power, coefficient in enumerate(poly):
        if (coefficient % (power + 1)) == 0:
            newCoefficient = coefficient // (power + 1)
        else:
            newCoefficient = coefficient / (power + 1)
        integral.append(newCoefficient)
    return integral
