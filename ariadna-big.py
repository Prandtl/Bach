import sys
import os

import re

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pylab

def getTestFuncs():
    def testFunction(point):
        x = point[0]
        y = point[1]
        x_sq = np.square(x)
        second_comp = np.square(x - y)
        result = x_sq + second_comp
        return result
    def testFunction2(point):
        x = point[0]
        y = point[1]
        x_sq = np.square(x)
        second_comp = np.square(x - y)
        result = x_sq + second_comp
        return result
    def testFunction3(point):
        x = point[0]
        y = point[1]
        result = 3 * np.power(x - 20, 2) + 20 * x * y - np.power(x * y, 3)
        return result
    def testFunction4(point):
        y = point[1]
        x = point[0]
        result = 20 + np.power(x, 2) - 10 * np.cos(2 * np.pi * x) + np.power(y, 2) - 10 * np.cos(2 * np.pi * y)
        return result
    return testFunction, testFunction2, testFunction3, testFunction4


if len(sys.argv) == 1:
    print("call ariadna with output filename like that:")
    print("\tpython ariadna.py example.out")
    exit()
filename = str(sys.argv[1])

if not os.path.isfile(filename):
    print("invalid filename")
    exit()

low = [-12, -12, -12, -12, -4, -4, -5, -5]
high = [12, 12, 12, 12, 25, 8, 5, 5]

xs = []
for i in range(0, 8):
    x = np.arange(low[i], high[i], 0.1)
    xs.append(x)

xx0, xx1 = np.meshgrid(xs[0], xs[1])
xx2, xx3 = np.meshgrid(xs[2], xs[3])
xx4, xx5 = np.meshgrid(xs[4], xs[5])
xx6, xx7 = np.meshgrid(xs[6], xs[7])
f1, f2, f3, f4 = getTestFuncs()


xxs=[xx0, xx1, xx2, xx3, xx4, xx5, xx6, xx7]
ravels = []
for xxi in xxs:
    ravels.append(np.ravel(xxi))

points = []
for i in range(0, 4):
    pts = list(zip(ravels[i*2], ravels[i*2 + 1]))
    points.append(pts)

zs1 = np.array([ f1(point) for point in points[0] ])
zs2 = np.array([ f2(point) for point in points[1] ])
zs3 = np.array([ f3(point) for point in points[2] ])
zs4 = np.array([ f4(point) for point in points[3] ])

zz1 = zs1.reshape(xx0.shape)
zz2 = zs2.reshape(xx2.shape)
zz3 = zs3.reshape(xx4.shape)
zz4 = zs4.reshape(xx6.shape)


plt.figure(1)
plt.subplot(221)
h1 = plt.contour(xx0, xx1, zz1, 40)
plt.subplot(222)
h1 = plt.contour(xx2, xx3, zz2, 40)
plt.subplot(223)
h1 = plt.contour(xx4, xx5, zz3, 40)
plt.subplot(224)
h1 = plt.contour(xx6, xx7, zz4, 40)

# f = open(filename, 'r')
# vec = []
# vecs = []
# for line in f:
#     trimmed = line.strip()
#     if re.match("^Process",trimmed):
#         continue
#     if re.match("^[-+]?[0-9]*\.?([0-9]+)?([eE][-+]?[0-9]+)?$", trimmed) is None:
#         if(len(vec)!=0):
#             vecs.append(vec)
#             vec=[]
#     else:
#         number = float(trimmed)
#         vec.append(number)
#
# x_list = [x for [x, y] in vecs]
# y_list = [y for [x, y] in vecs]
#
# plt.plot(x_list, y_list, 'ro', x_list, y_list, 'k')
plt.show()
