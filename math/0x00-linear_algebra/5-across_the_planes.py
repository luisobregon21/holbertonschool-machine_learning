#!/usr/bin/env python3
''' Add two matrices of integers '''


def add_matrices2D(mat1, mat2):
    ''' Add two matrices of integers '''

    sum = []

    if len(mat1[0]) != len(mat2[0]):
        return None

    for idx in range(len(mat1)):
        newList = []
        for idx2 in range(len(mat1[idx])):
            newList.append((mat1[idx][idx2] + mat2[idx][idx2]))
        sum.append(newList)
    return sum
