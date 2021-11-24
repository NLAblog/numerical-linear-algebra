# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 00:33:06 2021

@author: zahra
"""

import numpy as np
import numpy.linalg as la


print("LINALG LIBRARY OF NUMPY")
M = np.array([[3, 0, 2],
              [2, 0, -2],
              [0, 1, 1]])

print("Matrix M")
print("Determinant of M", np.linalg.det(M))
print("Inverse of M", np.linalg.inv(M))

def AddVectors(v1, v2):
    res= v1+v2
    return res

my_v1 = np.array([2,2])
my_v2 = np.array([1,3])

my_result =  AddVectors(my_v1, my_v2)
print(my_result)


A = np.arange(10).reshape((5,2))
print(A.shape)

print("######################################################")
print("FOR LOOPS")
i=0
for rows in A: 
    for cols in A:
        i=i+2
        print("Cols", cols)
    print(rows)
    
print("######################################################")
print("IF/THEN ELSE")
x= 0.6
if x >0.5:
    print("Its a head")
elif x<0.5:
    print("Its a tail")
else: 
    print("Its a tie")
    
import matplotlib.pyplot as plt
x = [0, 0.5, 2]
y = [0, 1, -1]
plt.plot(x, y, "r*")
plt.show()






