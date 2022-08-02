#!/usr/bin/env python3
''' plots a stacked bar graph'''
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

people = ['Farrah', 'Fred', 'Felicia']
fruits = {
    'apples': 'red',
    'bananas': 'yellow',
    'oranges': '#ff8000',
    'peaches': '#ffe5b4'
}

amount = len(people)
idx = 0
for fruitName, color in sorted(fruits.items()):
    bottom = 0
    for idx2 in range(idx):
        bottom += fruit[idx2]
    plt.bar(
        np.arange(amount),
        fruit[idx],
        width=0.5,
        bottom=bottom,
        color=color,
        label=fruitName
    )
    idx += 1

plt.xticks(np.arange(amount), people)
plt.yticks(np.arange(0, 81, 10))
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.legend()
plt.show()
