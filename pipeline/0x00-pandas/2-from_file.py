#!/usr/bin/env python3
''' from file'''
import pandas as pd


def from_file(filename, delimiter):
    '''
    loads data from a file as a pd.DataFrame
    :filename: file to load from
    :delimiter: column separator
    Returns: loaded pd.DataFrame
    '''
    return pd.read_csv(filename, sep=delimiter)
