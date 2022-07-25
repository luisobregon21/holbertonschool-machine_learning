#!/usr/bin/env python3
'''returns the transpose of a 2D matrix'''


def matrix_transpose(matrix):
    '''returns the transpose of a 2D matrix'''
    return [list(row) for row in zip(*matrix)]
