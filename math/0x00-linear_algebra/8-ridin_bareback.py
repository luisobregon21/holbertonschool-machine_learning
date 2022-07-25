#!/usr/bin/env python3


def mat_mul(mat1, mat2):
    ''' function multiplies 2 matrices '''
    product = []

    if len(mat1[0]) != len(mat2):
        return None

    # iterating by row of mat1
    for idx in range(len(mat1)):
        product.append([])
        # iterating by column by mat2
        for j in range(len(mat2[0])):
            # iterating by rows of mat2
            product[idx].append(0)
            for k in range(len(mat2)):
                product[idx][j] += mat1[idx][k] * mat2[k][j]

    return product
