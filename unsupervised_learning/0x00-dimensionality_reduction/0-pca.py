#!/usr/bin/env python3
''' perform PCA on a dataset '''


def pca(X, var=0.95):
    '''
    performs PCA on a dataset
    :X: is a numpy.ndarray of shape (n, d) where:
        n: is the number of data points
        d: is the number of dimensions in each point
        all dimensions have a mean of 0 across all data points
    :var: is the fraction of the variance that the
    PCA transformation should maintain

    Returns: the weights matrix, W, that
    maintains var fraction of X‘s original variance
    '''
