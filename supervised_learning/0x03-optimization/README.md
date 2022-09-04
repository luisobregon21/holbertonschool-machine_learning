# 0x03 optimization

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

## General

What is a hyperparameter?

- a hyperparameter is a parameter whose value is used to control the learning process.
- learning rate and batch size as well as mini-batch size are examples of hyperparameter.

How and why do you normalize your input data?

- generally performed during the data preprocessing step.
- the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization.
- Gradient descent converges much faster with feature scaling than without it.
- Basically for optimization.
  - if regularization is used as part of the loss function (so that coefficients are penalized appropriately)
  > any algorithm that computes distance or assumes normality, scale your features!!!
  >[When to scale](https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e)

- By normalizing all inputs to a standard scale, we're allowing the network to more quickly learn the optimal parameters for each input node.
- ensure that our inputs are roughly in the range of -1 to 1 to avoid weird mathematical artifacts associated with floating point number precision.
- computers lose accuracy when performing math operations on really large or really small numbers.
![optimization](https://www.jeremyjordan.me/content/images/2018/01/Screen-Shot-2018-01-24-at-1.12.52-PM.png)

What is a saddle point?

- It’s where the values of a parameter appear to have stopped converging toward some goal value because the values have switched from going down to going up briefly, and if you were to keep checking further along the curve the values would start going down again.
- Saddle points are a type of optimum combination of minima and maxima.
- When optimizing neural networks or any high dimensional function, the critical points(the points where the derivative is zero or close to zero) are saddle points. Saddle points, unlike local minima, are easily escapable.
![saddle points](https://qph.cf2.quoracdn.net/main-qimg-366252b64600a7d81446a83eb19e1e4a-pjlq)

### Gradient Decent

Gradient descent is the preferred way to optimize neural networks and many other machine learning algorithms but is often used as a black box.

- computes the gradient of the cost function w.r.t. to the parameters θ for the entire training dataset
- batch gradient descent can be very slow and is intractable for datasets that don't fit in memory.

```python
for i in range(nb_epochs):
  params_grad = evaluate_gradient(loss_function, data, params)
  params = params - learning_rate * params_grad
```

What is stochastic gradient descent?

- Stochastic gradient descent (SGD) in contrast performs a parameter update for each training example x^(i) and label y^(i)
- SGD does away with this redundancy by performing one update at a time.
- SGD performs frequent updates with a high variance that cause the objective function to fluctuate heavily as in Image 1.
- SDG keeps overshooting.

```python
for i in range(nb_epochs):
  np.random.shuffle(data)
  for example in data:
    params_grad = evaluate_gradient(loss_function, example, params)
    params = params - learning_rate * params_grad
```

What is mini-batch gradient descent?

- performs an update for every mini-batch of n training examples.
- Common mini-batch sizes range between 50 and 256, but can vary for different applications.
- the algorithm of choice when training a neural network.

```python
for i in range(nb_epochs):
  np.random.shuffle(data)
  for batch in get_batches(data, batch_size=50):
    params_grad = evaluate_gradient(loss_function, batch, params)
    params = params - learning_rate * params_grad
```

What is a moving average? How do you implement it?

- a calculation to analyze data points by creating a series of averages of different subsets of the full data set.
- Given a series of numbers and a fixed subset size, the first element of the moving average is obtained by taking the average of the initial fixed subset of the number series. Then the subset is modified by "shifting forward"; that is, excluding the first number of the series and including the next value in the subset.

What is gradient descent with momentum? How do you implement it?

- Momentum is a method that helps accelerate SGD in the relevant direction

![SGD NoMomentum](https://ruder.io/content/images/2015/12/without_momentum.gif) ![SGD momentum](https://ruder.io/content/images/2015/12/with_momentum.gif)

- adds  a fraction γ of the update vector of the past time step to the current update vector.

What is RMSProp? How do you implement it?

- adaptive learning rate method
- divides the learning rate by an exponentially decaying average of squared gradients.
- γ to be set to 0.9, while a good default value for the learning rate η is 0.001.

What is Adam optimization? How do you implement it?

- Adaptive Moment Estimation computes adaptive learning rates for each parameter.
- keeps an exponentially decaying average of past gradients mt, similar to momentum.
- Adam works well in practice and compares favorably to other adaptive learning-method algorithms

What is learning rate decay? How do you implement it?

- Learning rate decay is a technique for training modern neural networks. It starts training the network with a large learning rate and then slowly reducing/decaying it until local minima is obtained. It is empirically observed to help both optimization and generalization.

What is batch normalization? How do you implement it?

- Accelerating Deep Network Training by Reducing Internal Covariate Shift.
- makes normalization a part of the model architecture and performing the normalization for each training mini-batch.
- By ensuring the activations of each layer are normalized, we can simplify the overall loss function topology.
![batch normalization](https://www.jeremyjordan.me/content/images/2018/01/Screen-Shot-2018-01-24-at-1.16.09-PM.png)
