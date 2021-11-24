# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 09:39:10 2021

@author: zahra
"""

import numpy as np
import matplotlib.pyplot as plt


tol = 1e-15
#Is it a zero?

def ClassicalGramSchmidt(M):
    #What is the size of Q, R?
    Q = np.empty(M.shape)
    R = np.zeros((M.shape[1], M.shape[1]))
    v = np.empty(M.shape[0])
    
    n = M.shape[1] # number of columns of M
    for i in range(n):
        
        v = M[:, i]
        for j in range(i):
           
            R[j, i] = Q[:,j] @ M[:, i]
            v = v - R[j, i]*Q[:, j]
            
        R[i, i] = np.linalg.norm(v)
        
        if R[i,i] > tol:
            Q[:, i]= v/R[i,i]
        else:
            print("division by zero!")
    return Q,R 
            


# Given Matrix
epsilon = 1e-6
res = 1+epsilon*epsilon 
A = np.array([[1, 1, 1], [epsilon, 0, 0], [0, epsilon, 0], [0, 0, epsilon]])
print(np.linalg.cond(A))
print("A\n", A)
Q, R = np.linalg.qr(A)
#print(Q, R)
Q1, R1 = ClassicalGramSchmidt(A) 
print("Q: \n", Q1)
print("R: \n", R1)

#A=QR
print("Reconstructed A\n", Q1@R1 )


#Do it yourself!
#Q2, R2 = ModifiedGS(A)