#!/usr/bin/env python3
''''plots all 5 previous graphs in one figure'''
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig = plt.figure()

# line graph
axis1 = fig.add_subplot(3, 2, 1)
axis1.plot(y0, '-r')
axis1.set_xlim((0, 10))
fig.suptitle('All in One')

# scatter plot
axis2 = fig.add_subplot(3, 2, 2)
axis2.scatter(x1, y1, c='m')
axis2.set_xlabel('Height (in)', fontsize='x-small')
axis2.set_ylabel('Weight (lbs)', fontsize='x-small')
axis2.set_title("Men's Height vs Weight", fontsize='x-small')

# x ↦ y as a line graph
axis3 = fig.add_subplot(3, 2, 3)
axis3.plot(x2, y2)
axis3.set_xlabel('Time (years)', fontsize='x-small')
axis3.set_ylabel('Fraction Remaining', fontsize='x-small')
axis3.set_title('Exponential Decay of C-14', fontsize='x-small')
axis3.set_yscale('log')
axis3.set_xlim((0, 28650))

# x ↦ y1 and x ↦ y2 line graphs
axis4 = fig.add_subplot(3, 2, 4)
axis4.plot(x3, y31, 'r--', label='C-14')
axis4.plot(x3, y32, 'g-', label='Ra-226')
axis4.set_xlabel('Time (years)', fontsize='x-small')
axis4.set_ylabel('Fraction Remaining', fontsize='x-small')
axis4.set_title('Exponential Decay of Radioactive Elements',
                fontsize='x-small')
axis4.legend()
axis4.set_xlim((0, 20000))
axis4.set_ylim((0, 1))

bins = []
for x in range(11):
    bins.append(x*10)
axis5 = fig.add_subplot(3, 1, 3)
axis5.hist(student_grades, bins=bins, edgecolor='black')
axis5.set_xlabel('Grades', fontsize='x-small')
axis5.set_ylabel('Number of Students', fontsize='x-small')
axis5.set_title('Project A', fontsize='x-small')
axis5.set_xlim(0, 100)
axis5.set_xticks(np.arange(0, 101, 10))
axis5.set_ylim(0, 30)

plt.tight_layout()
plt.show()
