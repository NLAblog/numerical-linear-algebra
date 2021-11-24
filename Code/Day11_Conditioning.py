# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:30:26 2021

@author: zahra
"""
import numpy as np
from numpy import linalg as la
a = np.array([[1, 0, -1], [0, 1, 0], [1, 0, 1]])
print(a)

print("Condition number of A in 2-norm", la.cond(a), la.cond(a, 2))
print("Condition number of A in Frobenius", la.cond(a, 'fro'))

Sigma = la.svd(a, compute_uv=0)
print("Sigma values", Sigma)
sigma_min = min(Sigma)
sigma_max = max(Sigma)
print(sigma_min, sigma_max)
print("Condition number of a matrix in 2-norm sense is the ration of largestvs smallest singular value")

#Root finding, IN GENERAL, is an ill-conditioned problem
#Eigenvalues requires finding roots of a polynomial
#Case 1: Non symmetric problem
A = np.array([[1, 1000.], [0, 1.]])
print("Condition number of A", la.cond(A))
print(A)

A_perturbed = np.array([[1, 1000.], [0.001, 1.]])

Eig_A = la.eigvals(A)
Eig_A_perturbed  = la.eigvals(A_perturbed)

print("Eigenvalues of A : ", Eig_A, "\nEigenvalues of perturbed A", Eig_A_perturbed)

#Case 1: Symmetric problem

A = np.array([[1, 1000.], [1000, 1.]])
print("Condition number of A", la.cond(A))
print(A)

A_perturbed = np.array([[1, 1000.], [1000+ 0.001, 1.]])

Eig_A = la.eigvals(A)
Eig_A_perturbed  = la.eigvals(A_perturbed)

print("Eigenvalues of A : ", Eig_A, "\nEigenvalues of perturbed A", Eig_A_perturbed)
