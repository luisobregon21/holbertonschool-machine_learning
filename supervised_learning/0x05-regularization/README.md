# 0x05-regularization

## Learning Objectives

![graph fitting](https://cdn.analyticsvidhya.com/wp-content/uploads/2018/04/Screen-Shot-2018-04-03-at-7.52.01-PM-e1522832332857.png)

while going towards the right, the complexity of the model increases such that the training error reduces but the testing error doesn’t...

![test error](https://cdn.analyticsvidhya.com/wp-content/uploads/2018/04/Screen-Shot-2018-04-04-at-2.43.37-PM-768x592.png)

### General

What is regularization? What is its purpose?

Neural networks are really complex which makes them prone to overfitting. This is where regulazation comes in ...

Regulazation is a technique to make slight modifications to the learning algorithm so that the model generalizes better, improving model performance on unseen data.

What are L1 and L2 regularization?

L1 and L2 are the most common type of regulazation which update the general cost function by adding the "regulazation term".

_Cost function = Loss (say, binary cross entropy) + Regularization term_

What is the difference between the two methods?

L2: ![L2](https://cdn.analyticsvidhya.com/wp-content/uploads/2018/04/Screen-Shot-2018-04-04-at-1.59.54-AM.png)

In L2, lambda is the regulazation parameter.It is the hyperparameter whose value is optimized for better results. L2 regularization is also known as weight decay as it forces the weights to decay towards zero (but not exactly zero).

L1: ![L1](https://cdn.analyticsvidhya.com/wp-content/uploads/2018/04/Screen-Shot-2018-04-04-at-1.59.57-AM.png)

L1, penalizes the absolute value and the weights may be reduced to 0.

What is dropout?

![dropout](https://cdn.analyticsvidhya.com/wp-content/uploads/2018/04/1IrdJ5PghD9YoOyVAQ73MJw.gif)

roduces very good results and is consequently the most frequently used regularization technique.

At every iteration, dropout, randomly selects some nodes and removes them along with their incoming and outgoing connection.

![before](https://cdn.analyticsvidhya.com/wp-content/uploads/2018/04/Screen-Shot-2018-04-03-at-11.50.02-PM.png)

![after](https://cdn.analyticsvidhya.com/wp-content/uploads/2018/04/Screen-Shot-2018-04-03-at-11.52.06-PM.png)

What is early stopping?

Early stopping is a kind of cross-validation strategy where we keep one part of the training set as the validation set. When we see that the performance on the validation set is getting worse, we immediately stop the training on the model.

![early stopping](https://cdn.analyticsvidhya.com/wp-content/uploads/2018/04/Screen-Shot-2018-04-04-at-12.31.56-AM.png)

What is data augmentation?

The simplest way to reduce overfitting is to increase the size of the training data. An example of data augmentation can be seen in the image below ...

![data augmentation](https://cdn.analyticsvidhya.com/wp-content/uploads/2018/04/Screen-Shot-2018-04-04-at-12.14.45-AM.png)

How do you implement the above regularization methods in Numpy? Tensorflow?

What are the pros and cons of the above regularization methods?
