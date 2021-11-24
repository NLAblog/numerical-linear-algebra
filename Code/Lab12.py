# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 09:25:17 2021

@author: zahra
"""

import numpy as np
import matplotlib.pyplot as plt

#THIS LAB IS ABOUT LEAST SQUARES PROBLEM AND POLYNOMIAL INTERPOLATION

m=100
order = 20
n= order

matrix_dim = (m, n)
start =0
end =1
x = np.linspace(start, end, m)
print(x)
np.random.seed(1)
y= np.linspace(start, end, m) + 0.2*np.abs(np.random.randn(m))
print(y)

A = np.vander(x, matrix_dim[1])
A = np.fliplr(A)
print(np.linalg.matrix_rank(A))
c = np.linalg.lstsq(A, y, rcond =None)
print("Coeff of polynomial\n ",c)
coefs, residuals, rank , singular_values = np.linalg.lstsq(A, y, rcond=None)
print("Sigma", singular_values)
#

interp_m = 50
interp_x = np.linspace(start, end, interp_m)
interp_y = np.zeros(interp_m)
for ind, ix in enumerate(interp_x):
    interp_y[ind] = np.sum(coefs * ix ** np.arange(0, n))

plt.figure()
plt.plot(x, y, '*r', label='data points')
plt.plot(interp_x, interp_y, '-b', label='interp line')
plt.xlabel('x value')
plt.ylabel('y value')
plt.show()
