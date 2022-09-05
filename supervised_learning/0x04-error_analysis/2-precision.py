#!/usr/bin/env python3
'''calculates the precision'''

import numpy as np


def precision(confusion):
    '''
    calculates the precision for each class in a confusion matrix
    :confusion: is a confusion numpy.ndarray of shape (classes, classes) where row indices
    represent the correct labels and column indices represent the predicted labels
        classes: is the number of classes
    '''
    TP = np.diag(confusion)  # correct positive prediction
    FP = np.sum(confusion, axis=0) - TP  # incorrect positive prediction
    PREC = TP / (TP + FP)
    return PREC
