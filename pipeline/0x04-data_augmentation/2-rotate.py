#!/usr/bin/env python3
"""Module rotate_image."""
import tensorflow as tf


def rotate_image(image):
    """
    Rotate an image by 90 degrees counter-clockwise.

    Args:
        image (tensor): image to rotate
    Returns:
        rotated image
    """
    return (tf.image.rot90(image))
