# 0x06 keras

![meme image](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/11/c48e37d9cda2293173b7.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220913%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220913T003436Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e7e64074c4edd7ba4a3b8e34b11eae95602534bd50b65074a21fc3f6163755df)

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

### General

What is Keras?

- Keras is an open-source software library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library.
- Keras is the high-level API of TensorFlow 2 for implementing neural networks.

What is a model?

- The models are used to define TensorFlow neural networks by specifying the attributes, functions, and layers you want.

How to instantiate a model (2 ways):

- create an instance of the class. 
- Or create an array of layers and pass it to the constructor of the model.

```python
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))
```

```python
# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
```

How to build a layer

- add a layer after initiating the class, where each new line is a layer. 
- What input shape it recieves 
- show the role from input to prediction

```python
model.add(keras.Input(shape=(4,)))
```

How to add regularization to a layer:

How to add dropout to a layer:

How to add batch normalization:

How to compile a model:
```python
model.compile(...)
```
How to optimize a model:

How to fit a model:
```python
history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)
```

How to use validation data:

How to perform early stopping:

How to measure accuracy:

```python
test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
```

How to evaluate a model:

```The Model class offers a built-in training loop (the fit() method) and a built-in evaluation loop (the evaluate() method).```

How to make a prediction with a model:

How to access the weights/outputs of a model:

What is HDF5?:

How to save and load a model’s weights, a model’s configuration, and the entire model:
