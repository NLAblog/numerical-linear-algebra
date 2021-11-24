import numpy as np
import numpy.linalg as la
import time
import matplotlib.pyplot as plt

########################################################
##################### QUESTION 1 #######################
########################################################


def ComponentWiseMVM (M, v, hideTime = True):
    
    rows = len(M)
    cols = len(M[0])
    vdim = len(v)
    
    if vdim != cols:
        return ("Incorrect Dimensions")
    
    M_new = np.zeros((rows, 1))
    
    start = time.time()
    
    for i in range(rows):
        for j in range(cols):
            M_new [i] = M_new [i] + M[i, j]*v[j]
    
    if not hideTime:
        print('Time taken: ', time.time() - start)
    
    return M_new
    

def LinCombinationMVM (M, v, hideTime = True):
    
    rows = len(M)
    cols = len(M[0])
    vdim = len(v)
    
    if vdim != cols:
        return ("Incorrect Dimensions")
    
    M_new = np.zeros((rows, 1))

    start = time.time()
    
    for i in range (rows):
        M_new [i] = np.dot(M[i], v)
        
    if not hideTime:
        print('Time taken: ', time.time() - start)
        
    return M_new


def ComponentWiseMMM (M1, M2, hideTime = True):
     
    dim_1 = M1.shape
    dim_2 = M2.shape
    

    if dim_2[0] != dim_1[1]:
        return ("Incorrect Dimensions")
    
    M_new = np.zeros((dim_1[0], dim_2[1]))
    
    start = time.time()
    
    for i in range(dim_1[0]):
        for j in range(dim_1[1]):
            for k in range(dim_2[1]):
                M_new [i, k] = M_new [i, k] + M1[i, j]*M2[j, k]
                
    if not hideTime:
        print('Time taken: ', time.time() - start)
            
    return M_new


def LinCombinationMMM (M1, M2, hideTime = True):
    
    dim_1 = M1.shape
    dim_2 = M2.shape
    

    if dim_2[0] != dim_1[1]:
        return ("Incorrect Dimensions")
    
    M_new = np.zeros((dim_1[0], dim_2[1]))
    
    start = time.time()
    
    for i in range(dim_1[0]):
        for j in range(dim_2[1]):
            M_new[i, j] = np.dot(M1[i], M2[:,j])
            
    if not hideTime:
        print('Time taken: ', time.time() - start)
            
    return M_new


##################### PART 1 #######################
print('#############################################')
print('\n Question 1 | Part 1 \n')

A = np.array([ [1, 2], [4, 5] ])
B = np.array([ [1, 3], [6, 7] ])
x = np.array([10, 20])

print('\n Matrix Vector Componentwise Mulltiplication \n')
print(ComponentWiseMVM(A, x))
print('\n Matrix Vector Multiplication by Linear Combination')
print(LinCombinationMVM(A, x))
print('\n Matrix Matrix Componentwise Mulltiplication \n')
print(ComponentWiseMMM(A, B))
print('\n Matrix Matrix Multiplication by Linear Combination')
print(LinCombinationMMM(A, B))

##################### PART 2 #######################
print('#############################################')
print('\n Question 1 | Part 2 \n')

Arand = np.random.rand(1000, 50)
xrand = np.random.rand(50, 1)
Brand = np.random.rand(50, 600)

print('\n Random - Matrix Vector Componentwise Mulltiplication \n')
ComponentWiseMVM(Arand, xrand, hideTime = False)
print('\n Random - Matrix Vector Multiplication by Linear Combination')
LinCombinationMVM(Arand, xrand, hideTime = False)
print('\n Random - Matrix Matrix Componentwise Mulltiplication \n')
ComponentWiseMMM(Arand, Brand, hideTime = False)
print('\n Random - Matrix Matrix Multiplication by Linear Combination')
LinCombinationMMM(Arand, Brand, hideTime = False)


########################################################
##################### QUESTION 2 #######################
########################################################


class ImageProcessing :
    
    def __init__(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        
        self.dv1 = v2 - v1
        self.dv2 = v2 - v3
        self.dv3 = v1 - v3
        
        self.myObject = np.vstack([self.v1.T, self.v2.T, self.v3.T])
        
        self.ObjectPlotter(self.myObject, self.myObject, title = 'Initial Objecct')
        
    def vectorTransformation(self, matrix):
        
        v1_new = LinCombinationMVM(matrix, self.v1)
        v2_new = LinCombinationMVM(matrix, self.v2)
        v3_new = LinCombinationMVM(matrix, self.v3)
        
        return v1_new, v2_new, v3_new
        
    def anticlockwise_rotation (self, angle, degree = False, plot = True):
            
        self.angle = angle

        if degree:
            self.rotationMatrix = np.array([[np.cos(angle*np.pi/180), -1*np.sin(angle*np.pi/180)], [np.sin(angle*np.pi/180), np.cos(angle*np.pi/180)]])
        
        else:
            self.rotationMatrix = np.array([[np.cos(angle), -1*np.sin(angle)], [np.sin(angle), np.cos(angle)]])
           
        self.rotate1, self.rotate2, self.rotate3 = self.vectorTransformation(self.rotationMatrix)
        self.rotatedObject = np.vstack([self.rotate1.T , self.rotate2.T , self.rotate3.T])
        
        if plot:
            self.ObjectPlotter(self.myObject, self.rotatedObject, title = "Anti clockwise Rotation of Triangle")
        
        return self.rotationMatrix
    
    def reflection_yaxis (self, plot = True):
        
        self.reflectionMatrix = np.array([[-1, 0], [0, 1]])
        self.ref1, self.ref2, self.ref3 = self.vectorTransformation(self.reflectionMatrix)
        self.reflectedObject = np.vstack([self.ref1.T , self.ref2.T , self.ref3.T])
        
        if plot:
            self.ObjectPlotter(self.myObject, self.reflectedObject, title = 'Reflection of triangle in y axis')
            
        return self.reflectionMatrix
    
    def convolution_rotate_reflect (self, angle , Degree = False, plot = True):
        
        self.anticlockwise_rotation(angle, degree = Degree, plot = False)
        self.reflection_yaxis(plot = False)
        
        self.covMatrix = LinCombinationMMM(self.reflectionMatrix, self.rotationMatrix)
        self.cov1, self.cov2, self.cov3 = self.vectorTransformation(self.covMatrix)
        self.covObject = np.vstack([self.cov1.T, self.cov2.T , self.cov3.T])
        
        if plot:
            self.ObjectPlotter(self.myObject, self.covObject, title = 'Rotation followed by Reflection')
        
        return self.covMatrix
              
    def ObjectPlotter (self, origanal, transformed, title = ''):
    
        o1 = plt.Polygon(origanal, closed = False, color = 'red')
        o2 = plt.Polygon(transformed, closed = False, color = 'blue')
        ax = plt.gca()
        ax.add_patch(o1)
        ax.add_patch(o2)
        ax.set_xlim(-6,6)
        ax.set_ylim(-6,6)
        ax.set_title(title)
        plt.scatter(origanal[:,0], origanal[:,1], color = 'black')
        plt.scatter(transformed[:,0], transformed[:,1], color = 'black')
        plt.show()

    def checkUnitary (self, Matrix):
        
        OriganalList = np.array([la.norm(self.dv1), la.norm(self.dv2), la.norm(self.dv3)])
        
        t1, t2, t3 = self.vectorTransformation(Matrix)
        
        TransformedList = np.array([la.norm(t1 - t2) , la.norm(t2 - t3) , la.norm(t1 - t3)])
        
        X = LinCombinationMMM(Matrix, Matrix.T)
        
        if X.all() == np.eye(2).all():
            print("Unitary Matrix : Orthogonal Matrix is the inverse of Matrix")
        
            if OriganalList.sort() == TransformedList.sort():
                print ("Verify : Distances preserves, and hence the triangles are congruent so angles are also preserved")

            return True
    
        return ("Not Unitary Matrix as Orthogonal Matrix is not the inverse of the Matix")
    
##################### PART 1 #######################
print('#############################################')
print('\n Question 2 | Part 1 \n')
 
p1 = np.array([0, 1])
p2 = np.array([1, 4])
p3 = np.array([3, 5])

myObject = ImageProcessing(p1, p2, p3)

##################### PART 2 #######################
print('#############################################')
print('\n Question 2 | Part 2 \n')

myObject.anticlockwise_rotation(135, degree = 'True')

##################### PART 3 #######################
print('#############################################')
print('\n Question 2 | Part 3 \n')

myObject.reflection_yaxis()

##################### PART 4 #######################
print('#############################################')
print('\n Question 2 | Part 4 \n')

myObject.convolution_rotate_reflect(135, Degree = True)

##################### PART 5 #######################
print('#############################################')
print('\n Question 2 | Part 5 \n')

myObject.checkUnitary(myObject.anticlockwise_rotation(135, degree = True, plot = False))

##################### PART 6 #######################
print('#############################################')
print('\n Question 2 | Part 6 \n')

myObject.checkUnitary(myObject.reflection_yaxis(plot = False))

##################### PART 7 #######################
print('#############################################')
print('\n Question 2 | Part 7 \n')

myObject.checkUnitary(myObject.convolution_rotate_reflect(135, Degree = True, plot = False))