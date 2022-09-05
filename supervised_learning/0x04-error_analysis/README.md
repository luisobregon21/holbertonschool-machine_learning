# 0x04 Error Analysis

![meme](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/11/e3786a3d84e36ff800d8.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220905%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220905T134644Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=98dd74a4d6a85893c49e2b817ccd88e13e7a7459e3517a4b39661238d0b2d38b)

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

### General

What is the confusion matrix?

- AKA error matrix.
- A confusion matrix is a technique for summarizing the performance of a classification algorithm.
- specific table layout that allows visualization of the performance of an algorithm.
- Each row of the matrix represents the instances in an actual class while each column represents the instances in a predicted class, or vice versa...

![confusion matrix](https://miro.medium.com/max/667/1*3yGLac6F4mTENnj5dBNvNQ.jpeg)

What is type I error? type II?

- type I error is the mistaken rejection of an actually true null hypothesis...
  - a "false positive" finding or conclusion
  - ej.an innocent person is convicted
- type II error is the failure to reject a null hypothesis that is actually false...
  - a "false negative" finding or conclusion
  - ej. a guilty person is not convicted

What is sensitivity? specificity? precision? recall?

> **NOTE**: False positive is to guess the one's that are wrong as correct. True positive is to guess the one's that are correct as being actually right.
> False negative is to miss the correct answers. True negatives is to exclude the ones that were wrong.

- Sensitivity and specificity mathematically describe the accuracy of a test which reports the presence or absence of a condition.
  - Individuals for which the condition is satisfied are considered "positive" and those for which it is not are considered "negative".
- **Sensitivity** (true positive rate) refers to the probability of a positive test, conditioned on truly being positive.
  - sensitivity is a measure of how well a test can identify true positives
- **Specificity** (true negative rate) refers to the probability of a negative test, conditioned on truly being negative.
  - specificity is a measure of how well a test can identify true negatives.
- precision and recall are performance metrics that apply to data retrieved from a collection, corpus or sample space.
- **Precision** is the fraction of relevant instances among the retrieved instances
  - positive predictive value
- **Recall** the fraction of relevant instances that were retrieved
  - aka sensitivity

![example1](https://en.wikipedia.org/wiki/File:Precisionrecall.svg)

![precision vs recall](https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/700px-Precisionrecall.svg.png)

What is an F1 score?

- F-score or F-measure is a measure of a test's accuracy.
- calculated from the precision and recall of the test
- the precision is the number of true positive results divided by the number of all positive results
  - including those not identified correctly
- the recall is the number of true positive results divided by the number of all samples that should have been identified as positive
- The highest possible value of an F-score is 1.0
- the lowest possible value is 0, if either the precision or the recall is zero

What is bias? variance?

- Bias is the difference between the actual value and the predicted value.
- Variance is the variation in the predicted value across different datasets.

What is irreducible error?

- the error that we can not remove with our model, or with any model.
- The error is caused by elements outside our control

What is Bayes error?

- Bayes error rate is the lowest possible error rate for any classifier of a random outcome and is analogous to the irreducible error.

How can you approximate Bayes error?

- Bayes Classifier is defined as: 𝑎𝑟𝑔𝑚𝑖𝑛𝑓=𝐶𝑜𝑠𝑡(𝑓)

[How to calculate bias and variance](http://rasbt.github.io/mlxtend/user_guide/evaluate/bias_variance_decomp/)

How to create a confusion matrix

- calculating a confusion Matrix:
  - You need a test dataset or a validation dataset with expected outcome values.
  - Make a prediction for each row in your test dataset.
  - From the expected outcomes and predictions count:
    - The number of correct predictions for each class.
    - The number of incorrect predictions for each class, organized by the class that was predicted.

- Organize the numbers as followed:
  - Expected down the side: Each row of the matrix corresponds to a predicted class.
  - Predicted across the top: Each column of the matrix corresponds to an actual class.

#### WHAT TO DO IN THE SITUATION

- High Bias, High Variance -> Train more, Try a different architecture, Build a deeper network
- High Bias, Low Variance -> Train more, Try a different architecture, Build a deeper network
- Low Bias, High Variance -> Try a different architecture, Get more data, Use regularization
- Low Bias, Low Variance -> Nothing
