#!/usr/bin/env python3
"""Module flip_image."""
import tensorflow as tf


def flip_image(image):
    """
    Flip an image horizontally.

    Args:
        image (tensor): image to flip
    Returns:
        flipped image
    """
    return (tf.image.flip_left_right(image))
