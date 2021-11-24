import numpy as np
from matplotlib import pyplot
import numpy.linalg as la

if __name__ == '__main__':
    A = np.array([[3,4,4], [4, 3, -4]])
    print(A)
    print(A.shape)
    print("Rank", la.matrix_rank(A))
    
    U, Sigma, Vt = np.linalg.svd(A, full_matrices=True)
    print("U (True):\n", U)
    print("Sigma:\n", Sigma)
    print("Vt (True):\n", Vt)
    
    print(np.diag(Sigma))
    A_recreated = np.dot((np.dot(U, np.diag(Sigma))), Vt)
    print(A_recreated)
    
    pyplot.subplot(321)
    pyplot.imshow(A)
    pyplot.subplot(322)
    pyplot.imshow(A)
    pyplot.show()
    
    
    