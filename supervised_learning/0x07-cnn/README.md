# 0x07 CNN

A convolutional neural network (CNN, or ConvNet) is a class of artificial neural network (ANN), most commonly applied to analyze visual imagery.

> CNNs are regularized versions of multilayer perceptrons. Multilayer perceptrons usually mean fully connected networks, that is, each neuron in one layer is connected to all neurons in the next layer.

A Convolutional Neural Network is an artificial neural network mainly used for analyzing images. It can also be used for other data analysis or classification problems. CNN's are able to pick up/detect patterns to make sense of them. These patterns can be edges, faces, animals, etc. The difference between a regular Neural Network and a CNN is that a CNN has convolutional layers.

![Conv layer](https://miro.medium.com/max/1400/1*OHifHVQLIIumP865ASipXA.png)

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

### General

What is a convolutional layer?

These layers receive input and transform that input in the same way. Then passes transform input to the next layer. With each convolutional layer, the number of filter each layer should have needs to be specified. The filters are the ones that identify the patterns in a CNN.

> patterns: edges, shapes, textures, objects ... etc. That is usually identified at the start of the network.
> The deeper the network goes, the more sophisticated the filters become. Those filters might be able to detect specific objects: eyes, fur, feathers...
>> In more deeper layers they can detect  more so sophisticated objects like cats, dogs, birds...

![convolutional layer](https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_02_17A-ConvolutionalNeuralNetworks-WHITEBG.png)

![pattern recognition](https://miro.medium.com/max/1400/1*vkQ0hXDaQv57sALXAJquxA.jpeg)

What is a pooling layer?

Pooling layer operates on each feature map independently.

A pooling layer is another building block of a CNN. Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network.

![pooling](https://miro.medium.com/max/1100/1*gags_WLu961iw6I0ZX6iQA.png)

Most common approach is max pooling:
![max pooling](https://miro.medium.com/max/1400/1*jU_Mp73fXzh9_ffvtnbrDQ.png)

Forward propagation over convolutional and pooling layers

![Convolution Operation (Forward Pass)](https://miro.medium.com/max/1400/1*wqZ0Q4mBaHKjqWx45GPIow.gif)

Back propagation over convolutional and pooling layers

> Each weight in the filter contributes to each pixel in the output map. Thus, any change in a weight in the filter will affect all the output pixels.

![Derivative Computation (Backward pass)](https://miro.medium.com/max/1400/1*CkzOyjui3ymVqF54BR6AOQ.gif)

> At the pooling layer, forward propagation results in an N×N pooling block being reduced to a single value - value of the “winning unit”. Backpropagation of the pooling layer then computes the error which is acquired by this single value “winning unit”.

- Max-pooling - the error is just assigned to where it comes from - the “winning unit” because other units in the previous layer’s pooling blocks did not contribute to it hence all the other assigned values of zero

- Average pooling - the error is multiplied by 1/(N×N)
 and assigned to the whole pooling block (all units get this same value).

How to build a [CNN](https://www.tensorflow.org/tutorials/images/cnn#evaluate_the_model) using Tensorflow and Keras.

```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
```
