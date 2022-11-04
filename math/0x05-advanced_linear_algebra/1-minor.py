#!/usr/bin/env python3
''' calculates the Minor of a matrix '''

determinant = __import__('0-determinant').determinant


def minor_val(matrix, idx_r, idx_c):
    """
    Computes minor in each idx position of the given matrix
    :matrix: given matrix
    :idx_r: row skipped
    :idx_c: col skipped
    """
    minor_mat = [rows[:idx_c] + rows[idx_c + 1:]
                 for rows in (matrix[:idx_r] + matrix[idx_r + 1:])]
    return determinant(minor_mat)


def minor(matrix):
    """
    Compute the minor of a given matrix
    :matrix: list of lists whose determinant should be calculated
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
    if not all([len(mat) == mat_l for mat in matrix]):
        raise ValueError("matrix must be a non-empty square matrix")
    if mat_l == 1:
        return [[1]]

    minor_values = []
    for row in range_mat_l:
        minor_r = []
        for col in range_mat_l:
            minor_c = minor_val(matrix, row, col)
            minor_r.append(minor_c)
        minor_values.append(minor_r)
    return minor_values