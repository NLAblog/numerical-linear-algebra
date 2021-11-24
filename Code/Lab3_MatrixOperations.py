# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 11:25:42 2021

@author: zahra
"""

import numpy as np
#linear algebra review
print("1. Arrays")
u= np.array([1, 2, 3])
print("Rank 1 array", u)
print("type of array", type(u))
print("shape of array", u.shape)
print("Elements of array", u[0], u[2])
u[0] = 4
print(u)

v = np.array([[1,2], [2, 3], [3,4]])
print("Column matrix\n", v)
print("Shape of v", v.shape)
print("Rank of matrix v", 2)

M = np.array([[1, 2, 3],             
              [4, 5, 6],
              [7, 8, 9]])

print("Matrix M", M)
print("Shape/dimension of M", M.shape)

#Creating an identity matrix of dimension d
d=5
I= np.eye(d)
print(I)

v1 = np.array([1, 2, 4])
v2 = np.array([4, 5, 6])
v3 = np.array([7, 8, 9])
M = np.vstack([v1, v2])
print(M, M.shape)
print(M[:,1:3])

#np.multiply is a component-wise multiplication
b = np.multiply(M, v1)
print("Matrix-vector multiplication = \n", b)
#??? What is np.dot?
b = np.dot(M, v1)

print("Matrix-vector M.v1 = \n", b)
print("Matrix transpose & shape", M.T, M.T.shape)


#


