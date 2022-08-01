#!/usr/bin/env python3
''' slices a matrix along specific axes '''


def np_slice(matrix, axes={}):
    '''
    function slices a matrix along specific axes.
    axes is a dictionary where the key is an axis to
    slice along and the value is a tuple representing
    the slice to make along that axis
    '''

    slices = [slice(None)] * (max(axes) + 1)
    for axis, val in axes.items():
        slices[axis] = slice(*val)
    return matrix[tuple(slices)]
