# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 09:29:18 2021

@author: zahra
"""

import numpy as np

#Converts 1d array to column vector
def column_convertor(x):
    x.shape = (1, x.shape[0])
    return x

def get_norm(x):
    return np.sqrt(np.sum(np.square(x)))

# Returns Householder matrix for vector v
def householder_transformation(v):
    size_of_v = v.shape[1]
    e1 = np.zeros_like(v)
    e1[0, 0] = 1
    vector = get_norm(v) * e1
    if v[0, 0] < 0:
        vector = - vector
    u = (v + vector).astype(np.float64)
    print("u", u)
    #the most dominating part of the algorithm
    H = np.identity(size_of_v) - ((2 * np.matmul(np.transpose(u), u)) / np.matmul(u, np.transpose(u)))
    return H

# Return Q and R matrices for iter number of iterations.
def qr_step_factorization(q, r, iter, n):
    v = column_convertor(r[iter:, iter])
    Hbar = householder_transformation(v)
    print("\n\nH_bar\n", Hbar)
    H = np.identity(n)
    print(Hbar)
    H[iter:, iter:] = Hbar
    r = np.matmul(H, r)
    q = np.matmul(q, H)
    return q, r

def MYQR_TEST(A):
  
    m = len(A)
    n = len(A[0])
    Q = np.identity(m)
    R = A.astype(np.float64)
    
    print("Q, R", Q, R)
    for i in range(min(m, n)): #n if m>=n
        # For each iteration, H matrix is calculated for (i+1)th row
        Q, R = qr_step_factorization(Q, R, i, m)
    min_dim = min(m, n)
    R = np.around(R, decimals=2)
    R = R[:min_dim, :min_dim]
    Q = np.around(Q, decimals=2)
    print('A after QR factorization')
    print('R matrix')
    print(R, '\n')
    print('Q matrix')
    print(Q)

    print ("In built QR factorization:")
    orthonormal, upper_triangle = np.linalg.qr(A, 'complete')
    print("Q matrix", orthonormal)
    print("R matrix", upper_triangle)

    #CAN YOU CHECK WHICH ONE IS FASTER? IN COMPUTING TIME TAKEN, MAKE SURE TO COMMENT OUT THE 'PRINT' STATEMENTS
    #? Did I beat the default numpy.linalg.qr() implementation? Probably not :(

if __name__ == "__main__":
    A = np.array([[1,2,3], [4, 5, 6], [7, 8, 7], [4, 2, 3], [4, 2, 2]])
    print('The Matrix A is \n', A, )
    MYQR_TEST(np.copy(A))
    
    Q, R = np.linalg.qr(A, 'complete')
    print("Q, R\n", Q,"\n", R)
    
    

