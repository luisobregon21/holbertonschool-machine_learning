#!/usr/bin/python3
'''
determines if all the boxes can be opened:

- n number of locked boxes in front of you
- Each box is numbered sequentially from 0 to n - 1
- each box may contain keys to the other boxes.
'''


def canUnlockAll(boxes):
    ''' method determines if all the boxes can be opened '''

    # boxes is a list of lists
    # A key with the same number as a box opens that box
    isBoxOpenDict = {}
    for num, box in enumerate(boxes):
        isBoxOpenDict[num] = False

    isBoxOpenDict[0] = True
    for num, box in enumerate(boxes):
        for key in box:
            if key is not num and key in isBoxOpenDict:
                isBoxOpenDict[key] = True

    return all(x is True for x in isBoxOpenDict.values())
