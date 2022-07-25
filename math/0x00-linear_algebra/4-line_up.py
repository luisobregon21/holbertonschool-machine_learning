#!/usr/bin/env python3
''' Add two arrays of integers '''


def add_arrays(arr1, arr2):
    '''
    Add two arrays of integers
    '''
    if len(arr1) != len(arr2):
        return None
    else:
        return [arr1[i] + arr2[i] for i in range(len(arr1))]
