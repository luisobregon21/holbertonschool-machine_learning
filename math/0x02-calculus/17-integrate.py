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

    if type(C) is int or type(C) is float:
        if poly == [0]:
            return [C]

        if C % 1 == 0:
            C = int(C)
        integral = [C]

        for power in range(len(poly)):
            if type(poly[power]) != int and type(poly[power]) != float:
                return None

            if (poly[power] / (power + 1)) % 1 == 0:
                integral.append(int(poly[power] / (power + 1)))
            else:
                integral.append(poly[power] / (power + 1))

        for power in range(len(integral) - 1, 0, -1):
            if integral[power] == 0:
                integral.pop()
            else:
                break
        return integral
    return None
