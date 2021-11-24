# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 09:30:37 2021

@author: zahra
"""

import numpy as np
from matplotlib import image
from matplotlib import pyplot

#QR factorization (default uses LAPACK)


A = np.array([[1, 2, 4],
               [0, 0,5],
               [0, 3, 6]])

print("Rank of A",  np.linalg.matrix_rank(A))

Q, R = np.linalg.qr(A)

print("Q matrix\n", Q)
print("R matrix\n", R)


epsilon = 1.E-03
# print(epsilon)
#
# print("1+epsilon-sqaured", 1+pow(epsilon, 2))
# epsilon= 1+pow(epsilon, 2)
m = np.array([[1, 1, 1],
                [epsilon, 0, 0],
                [0, epsilon, 0 ],
                [0, 0,  epsilon ]])
# # reduced, complete
orthonormal, upper_triangle = np.linalg.qr(m, 'reduced')
print("Q matrix", orthonormal)
print ("R matrix", upper_triangle)
m_recreated = np.dot(orthonormal,upper_triangle)
print("m-epsilon\n", m, m_recreated)