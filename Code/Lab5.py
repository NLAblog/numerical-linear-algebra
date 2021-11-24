# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:26:13 2021

@author: zahra
"""
import numpy as np
import numpy.linalg as la

x=np.arange(1,5)
A = np.array([[3,4,4], [4, 3, -4]])

print( "Norms", la.norm(x), la.norm(A))

#linear transformation to a geometrial object defined by n-d set of m vectors
class ImageProcessor: 
    def __init__(self, name="", X=[]):
        print("Function:ImageProcessor constructor")
        self.imageName = name
        self.image  = X
    def rotate(self, X=[]):
        #R is a 2x2 matrix for planar rotations
        print("Rotate matrix of size 2xm")
        # oif X is empty then use the original image
        #else use X
        X_new = X
        return X_new
    def reflect(self, X) :
        print("Function:reflect")
        X_new =[]
        return X_new
    def plotImage(self, X):
        print("Function: Plot Image")
    

X = np.array([[0, 0.5, 2], [0, 1, -1]])
myTriangle = ImageProcessor("Tiangle", X)
rotated_triangle = myTriangle.rotate()
print(rotated_triangle)
res =myTriangle.reflect(rotated_triangle)
    
    
    
    
    
    
    
    
    
    
    