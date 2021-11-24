# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
'''
vectors - strictly 1 dimensional arrays
matrices are multidimensional arrays
arrays
'''

my_array = [1,2,3]
print(my_array)

np.array(my_array)


my_matrix = [[1,2,3],[4,2,8]]
print(my_matrix)

print(np.array(my_matrix))


#generating arrays of zeros
print(np.zeros(10))
print(np.ones(7))

#returns evenly space values for a given interval
x = np.arange(0, 20, 0.1)
y = np.linspace(0, 5, 21)

print(x)
print(y)

#identity matrix
print(np.eye(8))
np.random.seed(42)
print(np.random.randn(4, 4))


my_array = np.zeros(3)
my_array[1] =10
print(my_array)

#this is only to generate 
arr = np.arange(0,11)
print(arr)
print(arr[0:5])
temp = np.copy(arr)
temp[0:5] = 4
print(arr)
print(temp)

arr_2d = np.array(([5,10,15],[20,25,30],[35,40,45]))
arr_2d[1]
arr_2d[1][0]
arr_2d[1,0]
