#!/usr/bin/env python3
'''calculates the sensitivity'''

import numpy as np


def sensitivity(confusion):
    '''
    calculates the sensitivity for each class in a confusion matrix
    :confusion: is a confusion numpy.ndarray of shape (classes, classes) where row indices
    represent the correct labels and column indices represent the predicted labels
        classes: is the number of classes
    '''
    TP = np.diag(confusion)  # correct positive prediction
    FN = np.sum(confusion, axis=1) - TP  # incorrect positive prediction
    SN = TP / (TP + FN)
    return SN
