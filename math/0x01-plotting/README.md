# 0x01 Plotting

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

## General

- What is a plot?
  - A plot is a graphical technique for representing a data set

- What is a scatter plot? line graph? bar graph? histogram?
  - **scatterplot**: type of plot or mathematical diagram using Cartesian coordinates to display values for typically two variables for a set of data
  ![scatterplot](https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Scatter_diagram_for_quality_characteristic_XXX.svg/440px-Scatter_diagram_for_quality_characteristic_XXX.svg.png)

  - **Line Graph**: type of chart which displays information as a series of data points called 'markers' connected by straight line segments.
  ![line graph](https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/ScientificGraphSpeedVsTime.svg/600px-ScientificGraphSpeedVsTime.svg.png)

  - **Bar Graph**: presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent.
  ![Bar Graoh](https://upload.wikimedia.org/wikipedia/commons/3/35/Human_losses_of_world_war_two_by_country.png)

  - **Histogram**: an approximate representation of the distribution of numerical data.
  ![Histogram](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Histogram_of_arrivals_per_minute.svg/440px-Histogram_of_arrivals_per_minute.svg.png)

- What is matplotlib?
  - **Matplotlib** is a plotting library for Python.

- How to plot data with matplotlib

  ``` python
  import matplotlib.pyplot as plt
  plt.plot([1, 2, 3, 4])
  plt.show()
  ```

- How to label a plot

  ``` python
  plt.xlabel('some numbers')
  plt.ylabel('some name')
  ```

- How to scale an axis

``` python
# Set x-axis range
matplotlib.pyplot.xlim()

# Set y-axis range
matplotlib.pyplot.ylim()
```

- How to plot multiple sets of data at the same time
  - A subplot () function is a wrapper function which allows the programmer to plot more than one graph in a single figure by just calling it once.

``` python
  # importing libraries
import matplotlib.pyplot as plt
import numpy as np
import math

# Get the angles from 0 to 2 pie (360 degree) in narray object
X = np.arange(0, math.pi*2, 0.05)

# Using built-in trigonometric function we can directly plot
# the given cosine wave for the given angles
Y1 = np.sin(X)
Y2 = np.cos(X)
Y3 = np.tan(X)
Y4 = np.tanh(X)

# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(2, 2)

# For Sine Function
axis[0, 0].plot(X, Y1)
axis[0, 0].set_title("Sine Function")

# For Cosine Function
axis[0, 1].plot(X, Y2)
axis[0, 1].set_title("Cosine Function")

# For Tangent Function
axis[1, 0].plot(X, Y3)
axis[1, 0].set_title("Tangent Function")

# For Tanh Function
axis[1, 1].plot(X, Y4)
axis[1, 1].set_title("Tanh Function")

# Combine all the operations and display
plt.show()
```

### Output

![output](https://media.geeksforgeeks.org/wp-content/uploads/20201219163929/Figure2.png)

## Requirements

Allowed editors: vi, vim, emacs

All your files will be interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.8)

Your files will be executed with numpy (version 1.19.2) and matplotlib (version 3.3.4)

All your files should end with a new line

The first line of all your files should be exactly #!/usr/bin/env python3

A README.md file, at the root of the folder of the project, is mandatory

Your code should use the pycodestyle style (version 2.6)

All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')

All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')

All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')

Unless otherwise noted, you are not allowed to import any module

All your files must be executable

The length of your files will be tested using wc
