# 0x03 Probability

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

### General

What is probability?

- Probability is the branch of mathematics concerning numerical descriptions of how likely an event is to occur, or how likely it is that a proposition is true.
- probability of an event is a number between 0 and 1
- 0 indicates impossibility of the event and 1 indicates certainty
    ![probability](https://onlinestatbook.com/2/probability/graphics/simple_prob.gif)

Basic probability notation

- U -> OR, whats the probability one event or another will happen
  > P(A or B) = P(A) + P(B) - P(A and B)
- n -> AND, whats the probaility both conditions will be true
  > If events are independent P(A and B) = P(A) x P(B)
- | -> GIVEN, whats the probability with the information given, given becomes the new denominator.
  > If Events A and B are not independent, then P(A and B) = P(A) x P(B|A).

What is independence? What is disjoint?

- **disjoint** is when the events don't happen at the same time. They are mutually exclusive, they can't be independent.

- **independce** is when one factor doesn't affect the other
- new information won't affect the likely hood of the other
  - to check: P(A and B) = P(A) x P(B)
  - P(A) = P(A | B)

What is a union? intersection?

What are the general addition and multiplication rules?

> **General Mulitplication**:  P(A ∩ B) = P(A) * P(B|A)
>
> **General Addition**: P(A U B) = P(A) + P(B) - P(A ∩ B)

What is a probability distribution?

- The mathematical function that gives the probabilities of occurrence of different possible outcomes for an experiment.

![probability distribution](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Standard_deviation_diagram_micro.svg/200px-Standard_deviation_diagram_micro.svg.png)

What is a probability distribution function? probability mass function?

- A probability distribution is a mathematical description of the probabilities of events, subsets of the sample space.

- Discrete variables are countable in a finite amount of time.
- Continuous Variables would (literally) take forever to count.

- PMF -> gives probability of each discrete outcome
  - discrete variable -> can only have specific outcomes
- Probability Density Functions -> focuses on continoius varibles
  - PCF in this case will tell how much of the distribution is to the left of a given value.
  - higher the gradient, higher the density

What is a cumulative distribution function?

- A function that maps from values to their cumulative probabilities.CDF(x) is the **fraction of the sample less than or equal to x.**
- The CDF is a function of x, where x is any value that might appear in the distribution.
- Sum all the probability till you get to the event given.

```python
def EvalCdf(t, x):
    count = 0.0
    for value in t:
        if value <= x:
            count += 1

    prob = count / len(t)
    return prob
```

- This function is almost identical to PercentileRank, except that the result is a probability in the range 0–1 rather than a percentile rank in the range 0–100.

What is a percentile?

- The "mean" is the "average" you're used to, where you add up all the numbers and then divide by the number of numbers.

![mean](https://www.gstatic.com/education/formulas2/443397389/en/mean.svg)

- variance is the expectation of the squared deviation of a random variable from its population mean or sample mean. Variance is a measure of dispersion, meaning it is a measure of how far a set of numbers is spread out from their average value.
  
  - Variance is how reliable something is.

#### Calculate Variance

Step 1: Find the mean
Step 2: Find each score’s deviation from the mean
Step 3: Square each deviation from the mean
Step 4: Find the sum of squares
Step 5: Divide the sum of squares by n – 1 or N

- The difference between “percentile” and “percentile rank” is that PercentileRank takes a value and computes its percentile rank in a set of values; Percentile takes a percentile rank and computes the corresponding value.

What is mean, standard deviation, and variance?

- The Standard Deviation is a measure of how spread out numbers are (read that page for details on how to calculate it).

- When we calculate the standard deviation we find that generally:
  - 68% of values are within 1 standard deviation of the mean
  - 95% of values are within 2 standard deviations of the mean
  - 99.7% of values are within 3 standard deviations of the mean
  ![one sd](https://www.mathsisfun.com/data/images/normal-distrubution-3sds.svg)

### Common probability distributions

- Many cases where the data tends to be around a central value with no bias left or right, and it gets close to a "Normal Distribution" like this:
![Normal Distribution](https://www.mathsisfun.com/data/images/normal-distribution-1.svg)
- The blue curve is a Normal Distribution.The yellow histogram shows some data that
follows it closely, but not perfectly (which is usual).

- The Normal Distribution has:

  - mean = median = mode
  - symmetry about the center
  - 50% of values less than the mean and 50% greater than the mean

![another Normal Distribution](https://www.mathsisfun.com/data/images/normal-distribution-2.svg)

## More info on probability

### Discrete Distribution

Discrete random variables are described with a probability mass function (PMF). A PMF maps each value in the variable’s sample space to a probability.

- One such PMF is the uniform distribution over n possible outcomes: P(x=x) = 1/n.
  - This reads as “The probability of x taking on the value x is 1 divided by the number of possible values”.

- uniform distribution -> each outcome is equally likely

#### Bernoulli distribution

specifies the probability for a random variable which can take on one of two values (1/0, heads/tails, true/false, rain/no rain, etc.).

PMF of a Bernoulli distribution is P(x) = {p if x =1, and 1-p if x=0}.

### Continuous Distributions

Continuous random variables are described by probability density functions (PDF).

Generally indicated the PDF for a random variable x as f(x). PDFs map an infinite sample space to relative likelihood values.

#### Gaussian (Normal) distribution

AKA the bell curve. The Gaussian distribution is parameterized by two values: the mean μ (mu) and variance σ² (sigma squared).

The mean specifies the center of the distribution, and the variance specifies the width of the distribution.

standard deviation σ, is just the square root of the variance.

indicate that x is a random variable:

![formula](https://miro.medium.com/max/618/1*b6yxU4ivLewyufRtxjpXOw.png)

PDF function:

![pdf function](https://miro.medium.com/max/1342/1*XhNxPXLsU3Q-BzA4bXzZew.png)

- left side: “The PDF of x given μ and σ²

![example image](https://miro.medium.com/max/1346/1*P1ho8E4v6MSfXCZ5zKNFNQ.png)

- In PDF at x = x is not the actual probability of x.
  - probability of x taking on any specific value is actually 0!

- The integral of the PDF over the sample space is 1.

![pdfformula](https://miro.medium.com/max/850/1*Xgsk3qAICdosV55UnASzaA.png)

- the area under the PDF represents the total probability of the Gaussian

- continuous random variable’s cumulative distribution function (CDF)

!(cdf)[https://miro.medium.com/max/1080/1*x8DdNm9OAbIufxKLtvO2XQ.png]

- for a given value x, we’re taking the integral of the PDF from negative infinity to that value. So F(x) gives us the the area under the PDF for the interval negative infinity to x.

- F(x) gives us P(x≤x).

- use the CDF to determine the probability of any given range [a,b] by noticing that P(a≤x≤b) = F(b)-F(a).
  - P(x=x) is equivalent to asking P(x≤x≤x) = F(x)-F(x) = 0.

![cdf of gaussien](https://miro.medium.com/max/1340/1*Qns3LMbAsx-B6DGQFU2SYg.png)

## Requirements

- Allowed editors: vi, vim, emacs
- All your files will be interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.8)
- All your files should end with a new line
- The first line of all your files should be exactly #!/usr/bin/env python3
- A README.md file, at the root of the folder of the project, is mandatory
- Your code should use the pycodestyle style (version 2.6)
- All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
- All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
- All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
- Unless otherwise noted, you are not allowed to import any module
- All your files must be executable
- The length of your files will be tested using wc

### Mathematical Approximations

For the following tasks, you will have to use various irrational numbers and functions. Since you are not able to import any libraries, please use the following approximations:

π = 3.1415926536

e = 2.7182818285

![formula](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2019/4/5e71204ca545072e8766.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220808%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220808T173152Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=360deba8d6ddbacee657eb16ce0b3abbfee5b1d4bcd5813dbb6fde230f00a5e8)
