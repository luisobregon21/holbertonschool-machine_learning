#!/usr/bin/env python3
''' function concatenates 2 matrices' along a specific axis '''


def cat_matrices2D(mat1, mat2, axis=0):
    ''' function concatenates 2 matrices' along a specific axis '''
    newMatrix = []
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        for row in mat1:
            newMatrix.append(row)
        for row in mat2:
            newMatrix.append(row)
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        for row1, row2 in zip(mat1, mat2):
            newMatrix.append(row1 + row2)
    return newMatrix
