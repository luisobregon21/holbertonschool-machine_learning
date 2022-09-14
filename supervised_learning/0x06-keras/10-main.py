#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
import tensorflow.keras as K
import tensorflow as tf
import numpy as np
import random
import os
SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Imports
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model
model = __import__('9-model')
weights = __import__('10-weights')

if __name__ == '__main__':

    network = model.load_model('network2.h5')
    weights.save_weights(network, 'weights2.h5')
    del network

    network2 = model.load_model('network1.h5')
    print(network2.get_weights())
    weights.load_weights(network2, 'weights2.h5')
    print(network2.get_weights())
