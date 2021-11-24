#BACKWARD STABILITY OF QR (using Householder Reflectors)
# The function under consideration is a matrix matrix multiplication, input is Q,R, output is A

import numpy as np
import numpy.linalg as la

#Lets create R (upper triangular matrix) using random numbers
np.random.seed(1)
R = np.random.random((50,50))
R *=np.tri(*R.shape)
R= R.T
print(R)

#Let's create a Q (randomly) using QR decomposition for some matrix Y
np.random.seed(2)
Y= np.random.random((50, 50))
Q, dummyX = la.qr(Y, 'complete')
print(Q.shape)

#Let's construct an A using Q, R as above this is our \tildef(x)
#matmul(,) is our \tildef, however treated as f (A is the basis of comparison, using QR)
A = np.matmul(Q, R)


#Let's factorize A using QR (householder reflectors again)
Q2, R2 = la.qr(A)
A_new =  np.matmul(Q2, R2) # this is our \f(\tildex)

#Question: Is Q2 = Q? Is R2 =R?
#Let's investigate
# ||x- \tildex||/ ||x||,  ||f(x)-f(\tildex)|| / ||f(x)||

error_abs_Q = la.norm(Q2-Q)
error_rel_Q = la.norm(Q2-Q)/la.norm(Q) #this is ||x1 - \tildex1||/||x1||
error_rel_R = la.norm(R2-R)/la.norm(R) # this is ||x2 - \tildex2||/||x2||
print("Absolute, Relative error for Q ", error_abs_Q, error_rel_Q)
print("Relative error for R ", error_rel_R)

#Backward error in Q2R2 is the residual/backward error
error_rel_A = la.norm(A - A_new )/(la.norm(A))
print("Relative error of A, ", error_rel_A) # we get machine level precision!
#We conclude that QR factorization is backward stable!

#So, what happens if you used a backward stable algorithm for a 'further perturbed input'
#Sensitivity analysis for perturbations in input.
#Now, let us perturb Q and R even further
Q3 = Q + 1.e-8*np.random.random((50,50)) # Q3 is not exactly orthogonal, there is 'some' loss of orthogonality
R3 = R + + 1.e-11*np.random.random((50,50)) # losing out on upper triangulation
error_rel_Q3 = la.norm(Q3-Q)/la.norm(Q)
error_rel_R3 = la.norm(R3-R)/la.norm(R)
#Notice that the errors in R3, Q3 do not tell us much about the loss in orthogonality and loss of upper triangulation.

#Starting with bigger ||x- \tildex||/ ||x||, we expect a certain error  (thats bigger than the one before) thats reflected in the computation of A
print("Absolute, Relative error for Q3,R3", error_rel_Q3, error_rel_R3 )
error_rel_A_perturbed = la.norm(A - np.matmul(Q3,R3) )/(la.norm(A))
print("Relative error of A", error_rel_A_perturbed)
#QR factorization is sensitive to 'loss of orthogonality' and to 'perturbations in R'
#Notice the drop in accuracy between A computed from Q3,R3 as compared to A original, even though the factorization algorithm is backward stable.
#The reason is because of the sensitivity of the input data, i.e. orthogonality, upper triangulation, and not because of the algorithm
#sensitivity to input perturbations is different from backward stability of the algorithm!
