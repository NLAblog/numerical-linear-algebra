# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:18:29 2021

@author: zahra
"""
import numpy as np
from matplotlib import pyplot
from matplotlib import image


    

def svd(X): 
    U, Sigma, Vh = np.linalg.svd(X,
      full_matrices=False, # It's not necessary to compute the full matrix of U or V
      compute_uv=True)
    
    return U, Sigma, Vh

def LossyCompression(U, Sigma, V, k):
    truncated_u = U[:, :k]
    truncated_v = V[:k, :]
    
    truncated_s = np.empty(k)
    
    truncated_s = Sigma[:k]
  
    return (truncated_u @ np.diag(truncated_s) @ truncated_v)


if __name__ == '__main__':
    data = image.imread('Coco.jpeg')
    print("Date type", data.dtype)
    print("Image Size", data.shape)
    # # reshaping image into 2D
    A = data[:, :, 0]
    print("New Image Size", A.shape)
    pyplot.imshow(A)
    pyplot.title("Original Coco's image")
    pyplot.show()
    #print(A)
    
    U, Sigma, Vh = svd(A)
    A_recreated = np.dot(np.dot(U, np.diag(Sigma)), Vh)
    
    pyplot.imshow(A_recreated)
    pyplot.title("Recreated Coco's image")
    pyplot.show()
    #print(A)
    
    rank =  np.linalg.matrix_rank(A)
    print("Rank  = ", rank )
    
    k = 20
    A_compressed  = LossyCompression(U, Sigma, Vh, k)
    print("Size of compressed image", A_compressed.shape)
    
    pyplot.imshow(A_compressed)
    pyplot.title("Recreated reduced rank Coco's image")
    pyplot.show()
    
    
    pyplot.subplot(323)
    pyplot.title("Fully recreated after SVD")
    pyplot.imshow(A)
    #
    pyplot.subplot(324)
    pyplot.title("Lossy_Compressed")
    pyplot.imshow(A_compressed)
    name = "Result_" + str(k) +".png"
    image.imsave(name, A_compressed)
    pyplot.show()
    
    #print(Sigma)
    
    
    
    
    