#!/usr/bin/env python3
''' performs pooling on images '''

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    '''
    performs a performs pooling on images
    :images: a numpy.ndarray with shape (m, h, w) containing
    multiple images with channels
        m: is the number of images
        h: is the height in pixels of the images
        w: is the width in pixels of the images
        c: is the number of channels in the image
    :kernel_shape: is a numpy.ndarray with shape
    (kh, kw) containing the kernel for the convolution
        kh: is the height of the kernel
        kw: is the width of the kernel
    :stride: is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    :mode: indicates the type of pooling
        max: indicates max pooling
        avg: indicates average pooling
    '''
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = int(((h - kh) / sh) + 1)
    ow = int(((w - kw) / sw) + 1)

    # convolutional dimension
    conv_image = np.zeros((m, oh, ow, c))
    # padded images
    for y in range(oh):
        for x in range(ow):
            image_slice = images[:, (y*sh):(y*sh)+kh, (x*sw):(x*sw)+kw, :]

            # change values of zero's array
            if mode == 'max':
                conv_image[:, y, x, :] = np.max(image_slice, axis=(1, 2))
            else:
                conv_image[:, y, x, :] = np.average(image_slice, axis=(1, 2))
    return conv_image
