#!/usr/bin/env python3
'''calculates the specificity'''

import numpy as np


def specificity(confusion):
    '''
    calculates the specificity for each class in a confusion matrix
    :confusion: is a confusion numpy.ndarray of shape (classes, classes) where row indices
    represent the correct labels and column indices represent the predicted labels
        classes: is the number of classes
    '''
    TP = np.diag(confusion)  # correct positive prediction
    FP = np.sum(confusion, axis=0) - TP  # incorrect positive prediction
    FN = np.sum(confusion, axis=1) - TP  # incorrect negative prediction
    TN = np.sum(confusion) - (FP + FN + TP)  # correct negative prediction

    SP = TN / (TN + FP)
    return SP
