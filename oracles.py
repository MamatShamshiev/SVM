import numpy as np
import scipy


class BinaryHinge():
    
    def __init__(self, C=1.0):
        self.C = C
     
    def func(self, X, y, w):
        """
        Считается, что нулевая координата вектора w соответсвует w_0, 
        в Х присутсвует единичный столбец.
        """
    
        M = 1 - y * X.dot(w)
        M[M < 0] = 0
        return (self.C / X.shape[0]) * M.sum() + 0.5 * w[1:].dot(w[1:].T)
        
    def grad(self, X, y, w):
        """
        Считается, что нулевая координата вектора w соответсвует w_0, 
        в Х присутсвует единичный столбец.
        """
        
        M = 1 - y * X.dot(w)
        reg_grad = np.copy(w)
        reg_grad[0] = 0
        return (- self.C / X.shape[0]) * X.T.dot(y * np.where(M > 0, 1, 0)) + reg_grad
