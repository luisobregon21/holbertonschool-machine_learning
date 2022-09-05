#!/usr/bin/env python3
'''calculates the F1 score'''

precision = __import__('2-precision').precision
sensitivity = __import__('1-sensitivity').sensitivity


def f1_score(confusion):
    '''
    calculates the F1 score of a confusion matrix
    :confusion: is a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels
        classes: is the number of classes
    '''
    prec = precision(confusion)
    recall = sensitivity(confusion)
    f1 = 2 * (prec * recall) / (prec + recall)
    return f1
