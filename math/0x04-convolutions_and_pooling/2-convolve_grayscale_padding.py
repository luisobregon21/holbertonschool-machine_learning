#!/usr/bin/env python3
''' custom padding convolution on grayscale images '''

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    '''
    performs a custom padding convolution on grayscale images
    :images: a numpy.ndarray with shape (m, h, w) containing
    multiple grayscale images
        m: is the number of images
        h: is the height in pixels of the images
        w: is the width in pixels of the images
    :kernel: is a numpy.ndarray with shape
    (kh, kw) containing the kernel for the convolution
        kh: is the height of the kernel
        kw: is the width of the kernel
    :padding: is a tuple of (ph, pw)
        ph: is the padding for the height of the image
        pw: is the padding for the width of the image
        the image should be padded with zeros
    '''
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    oh = h + 2 * ph - kh + 1
    ow = w + 2 * pw - kw + 1

    # convolutional dimension
    conv_image = np.zeros((m, oh, ow))
    # padded images
    padded_images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant')
    for y in range(oh):
        for x in range(ow):
            image_slice = padded_images[:, y:y+kh, x:x+kw]
            # change values of zero's array
            conv_image[:, y, x] = np.tensordot(image_slice, kernel)
    return conv_image
