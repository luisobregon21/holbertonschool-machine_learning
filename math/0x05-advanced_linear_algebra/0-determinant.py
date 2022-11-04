#!/usr/bin/env python3
''' calculates the determinant of a matrix '''


def find_determinant(matrix):
    '''
    omputes the determinant of a given matrix

    :matrix: list of lists whose determinant should be calculated
    Returns: matrix given to find it's determinant
    '''
    length = len(matrix)

    if length == 2 and len(matrix[0]) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    det = 0
    cols = list(range(len(matrix)))
    for col in cols:
        mat_cp = [r[:] for r in matrix]
        mat_cp = mat_cp[1:]
        rows = range(len(mat_cp))

        for row in rows:
            mat_cp[row] = mat_cp[row][0:col] + mat_cp[row][col + 1:]
        sign = (-1) ** (col % 2)
        sub_det = find_determinant(mat_cp)
        det += sign * matrix[0][col] * sub_det
    return det


def determinant(matrix):
    '''
    calculates the determinant of a matrix
    :matrix: matrix given to find it's determinant
    '''

    mat_len = len(matrix)
    if type(matrix) != list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all([type(mat) == list for mat in matrix]):
        raise TypeError("matrix must be a list of lists")
    if matrix[0] and mat_len != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    if matrix == [[]]:
        return 1
    if mat_len == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    if not all(mat_len == len(col) for col in matrix):
        raise ValueError("matrix must be a square matrix")

    return find_determinant(matrix)
