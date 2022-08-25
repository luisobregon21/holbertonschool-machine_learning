#!/usr/bin/env python3

import tensorflow.compat.v1 as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders

x, y = create_placeholders(784, 10)
print(x)
print(y)
