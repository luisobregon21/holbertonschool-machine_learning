#!/usr/bin/env python3
'''Inverse matrix'''

determinant = __import__('0-determinant').determinant
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """
    Calculates the inverse of a matrix
    :matrix: matrix whose inverse should be calculated
    """
    mat_l = len(matrix)

    if type(matrix) != list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all([type(m) == list for m in matrix]):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    if matrix[0] and len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    if not all(mat_l == len(col) for col in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if mat_l == 1 and len(matrix[0]) == 1:
        return [[1/(matrix[0][0])]]
    if mat_l == 1:
        if matrix[0][0] == 0:
            return None
    if determinant(matrix) == 0:
        return None

    det = determinant(matrix)
    adjugate_mat = adjugate(matrix)
    inversed = [[mat_minor / det for mat_minor in row]
                for row in adjugate_mat]
    return inversed
