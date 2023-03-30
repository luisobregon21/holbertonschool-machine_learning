#!/usr/bin/env python3
"""Module change_hue."""
import tensorflow as tf


def change_hue(image, delta):
    """
    Change the hue of an image.

    Args:
        image (tensor): contains the image to change
        delta (float): amount the hue should change
    Returns:
        altered image
    """
    return (tf.image.adjust_hue(image, delta))
