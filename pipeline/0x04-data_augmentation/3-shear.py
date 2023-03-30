#!/usr/bin/env python3
"""Module shear_image."""
import tensorflow as tf


def shear_image(image, intensity):
    """
    Shear an image randomly.

    Args:
        image (tensor): image to shear
        intensity (int): intensity with which the image should be sheared
    Returns:
        sheared image
    """
    shear = tf.keras.preprocessing.image.random_shear(
        image, intensity)
    return shear
