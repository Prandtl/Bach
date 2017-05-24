import sys
import os

import re

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pylab

def getTestData(h):
    def testFunction(point):
        x = point[0]
        y = point[1]
        x_sq = np.square(x)
        second_comp = np.square(x - y)
        result = x_sq + second_comp
        return result
    def numerical_diff_x(point):
        #(f2-f0)/2h
        x = point[0]
        y = point[1]
        f0 = testFunction((x - h, y))
        f2 = testFunction((x + h, y))
        dx = (f2 - f0)/(2 * h)
        return dx
    def numerical_diff_y(point):
        #(f2-f0)/2h
        x = point[0]
        y = point[1]
        f0 = testFunction((x, y - h))
        f2 = testFunction((x, y + h))
        dy = (f2 - f0)/(2 * h)
        return dy
    def getGradientIn(point):
        return np.array([numerical_diff_x(point), numerical_diff_y(point)])
    def getAntiGradientIn(point):
        g = getGradientIn(point)
        return -getGradientIn(point)
    return testFunction, getGradientIn, getAntiGradientIn


if len(sys.argv) == 1:
    print("call ariadna with output filename like that:")
    print("\tpython ariadna.py example.out")
    exit()
filename = str(sys.argv[1])

if not os.path.isfile(filename):
    print("invalid filename")
    exit()

x = np.arange(-12, 12, 0.1)
y = np.arange(-12, 12, 0.1)
xx, yy = np.meshgrid(x, y)
f, g, a = getTestData(1e-10)
xravel = np.ravel(xx)
yravel = np.ravel(yy)
points = list(zip(xravel, yravel))
z = np.array([ f(point) for point in points ])
zz = z.reshape(xx.shape)
h = plt.contour(xx, yy, zz, 40)

f = open(filename, 'r')
vec = []
vecs = []
for line in f:
    trimmed = line.strip()
    if re.match("^Process",trimmed):
        continue
    if re.match("^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$", trimmed) is None:
        if(len(vec)!=0):
            vecs.append(vec)
            vec=[]
    else:
        number = float(trimmed)
        vec.append(number)

x_list = [x for [x, y] in vecs]
y_list = [y for [x, y] in vecs]

plt.plot(x_list, y_list, 'ro', x_list, y_list, 'k')
plt.show()
