#!/usr/bin/env python3
''' valid convolution on grayscale images '''

import numpy as np


def convolve_grayscale_valid(images, kernel):
    '''
    performs a valid convolution on grayscale images
    :images: a numpy.ndarray with shape (m, h, w) containing
    multiple grayscale images
        m: is the number of images
        h: is the height in pixels of the images
        w: is the width in pixels of the images
    :kernel: is a numpy.ndarray with shape
    (kh, kw) containing the kernel for the convolution
        kh: is the height of the kernel
        kw: is the width of the kernel
    '''
    m, h, w = images.shape
    kh, kw = kernel.shape
    conv_w = (w - kw) + 1
    conv_h = (h - kh) + 1

    conv_image = np.zeros((m, conv_w, conv_h))

    for y in range(conv_h):
        for x in range(conv_w):
            image_slice = images[:, y:y+kh, x:x+kw]
            conv_image[:, y, x] = np.tensordot(image_slice, kernel)
    return conv_image
