# 0x08. Deep Convolutional Architectures

## 1x1 Convolution

- looks at 1 pixel instead of the patch of the image.

- regular convolutional is basically a linear classifier for the patch of the image.

- adding a 1 by 1 convolution in the middle of the convolution, that is basically like having a mini NN running over the patch instead of linear classifier.

- basically matrix multiply with few parameters.

- makes model deeper and have more parameters without changing the the structure.

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

## General

What is a skip connection?

What is a bottleneck layer?

What is the Inception Network?
 Inception architecture, we use the 1x1 convolutional filters to reduce dimensionality in the filter dimension. These 1x1 conv layers can be used in general to change the filter space dimensionality (either increase or decrease) and in the Inception architecture we see how effective these 1x1 filters can be for dimensionality reduction, explicitly in the filter dimension space, not the spatial dimension space.

What is ResNet? ResNeXt? DenseNet?

How to replicate a network architecture by reading a journal article
