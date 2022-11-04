#!/usr/bin/env python3
'''matrix of cofactors'''

minor_val = __import__('1-minor').minor_val


def cofactor(matrix):
    """
    Compute the matrix of cofactors
    :matrix: matrix whose cofactors should be calculated
    """
    mat_l = len(matrix)
    range_mat_l = range(len(matrix))

    if type(matrix) != list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all([type(mat) == list for mat in matrix]):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    if matrix[0] and len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    if not all(mat_l == len(col) for col in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if mat_l == 1 and len(matrix[0]) == 1:
        return [[1]]
    if mat_l == 2 and len(matrix[0]) == 2:
        return [[matrix[1][1], -matrix[1][0]], [-matrix[0][1], matrix[0][0]]]

    minor_values = []
    for row in range_mat_l:
        minor_r = []
        for col in range_mat_l:
            minor_c = minor_val(matrix, row, col)
            sign = (-1) ** (row + col)
            minor_r.append(minor_c * sign)
        minor_values.append(minor_r)
    return minor_values
