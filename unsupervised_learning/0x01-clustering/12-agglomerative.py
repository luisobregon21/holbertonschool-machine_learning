#!/usr/bin/env python3
"""Module agglomerative."""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Perform agglomerative clustering on a dataset.

    :X (ndarray): contains the dataset
    :dist (int): maximum cophenetic distance for all clusters

    Returns:
        clss, contains the cluster indices
    """
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(linkage, t=dist,
                                            criterion='distance')
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()

    return clss
