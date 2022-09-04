#!/usr/bin/env python3
'''calculates the weighted moving average'''


def moving_average(data, beta):
    '''
    calculates the weighted moving average of a data set
    :data: list of data to calculate the moving average of
    :beta: weight used for the moving average
    Returns: a list containing the moving averages of data
    '''
    vt = 0
    moving_average = []
    for idx in range(len(data)):
        vt = beta * vt + (1 - beta) * data[idx]
        bias_correction = 1 - beta ** (idx + 1)
        moving_average.append(vt / bias_correction)
    return moving_average
