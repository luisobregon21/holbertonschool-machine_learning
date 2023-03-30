#!/usr/bin/env python3
"""Module change_brightness."""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Change the brightness of an image randomly.

    Args:
        image (tensor): contains the image to change
        max_delta (float): maximum amount the image should be brightened (or
                           darkened)
    Returns:
        altered image
    """
    return (tf.image.random_brightness(image, max_delta))
