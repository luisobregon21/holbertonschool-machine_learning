#!/usr/bin/env python3
"""Module crop_image."""
import tensorflow as tf


def crop_image(image, size):
    """
    Perform a random crop of an image.

    Args:
        image (tensor): image to crop
        size (tuple): size of the crop
    Returns:
        cropped image
    """
    return (tf.image.random_crop(image, size=size))
