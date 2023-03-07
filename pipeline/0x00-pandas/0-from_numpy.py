#!/usr/bin/env python3
''' from numpy '''
import numpy as np
import pandas as pd

def from_numpy(array):
    '''
    creates a pd.DataFrame from a np.ndarray
    :array: is the np.ndarray from which you should create the pd.DataFrame
    Returns: the newly created pd.DataFrame
    '''
    columns = list('ABCDEFGH')
    reshape = columns[:array.shape[1]]
    return pd.DataFrame(array, columns=reshape)
