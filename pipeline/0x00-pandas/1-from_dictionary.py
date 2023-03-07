#!/usr/bin/env python3
''' created a pd.DataFrame from dictionary '''
import pandas as pd


col = {'First': [0.0, 0.5, 1.0, 1.5],
       'Second': ['one', 'two', 'three', 'four']}
df = pd.DataFrame(col, index=list('ABCD'))
