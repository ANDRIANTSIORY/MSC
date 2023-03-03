import numpy as np
import scipy.linalg as la

class Generate_tensor():
    def __init__(self, m, n1, n2, k1=1, k2=1, k3=0, sigma = 1):
        # Tensor of dimension (m, n1, n2)
        # k1 = |J_1|, k2 = |J_2| and k3 = |J_3|
        # signal strength : sigma
        self._m = m
        self._n1 = n1
        self._n2 = n2
        self._k1 = k1
        self._k2 = k2
        self._k3 = k3
        self._sigma = sigma

    def rank_one(self):
        # The true cluster
        J1_true = list(range(0,self._k1))
        J2_true = list(range(0, self._k2))
        J3_true = list(range(0, self._k3))
        # create a random vector 
        v = np.zeros((self._m))
        u = np.zeros((self._n1))
        w= np.zeros((self._n2))
        k = max([self._k1, self._k2, self._k3])
        k = 1/k
        X = np.zeros((self._m,self._n1,self._n2))
        for i in range(self._m):
            if i in J3_true:
                v[i] = 1/np.sqrt(self._k3)
            else:
                v[i] = np.random.normal(0, k)
        for i in range(self._n1):
            if i in J1_true:
                u[i] = 1/np.sqrt(self._k1)
            else:
                u[i] = np.random.normal(0, k)
        for i in range(self._n2):
            if i in J2_true:
                w[i] = 1/np.sqrt(self._k2)
            else:
                w[i] = np.random.normal(0,k)
        for i in range(self._m):
            for j in range(self._n1):
                for k in range(self._n2):
                    X[i,j,k] = v[i]*u[j]*w[k]
                    if (i in J3_true ) and (j in J1_true) and (k in J2_true):
                        X[i,j,k] = self._sigma*X[i,j,k]

        Z = np.random.normal(0, 1,  size= (self._m, self._n1, self._n2))
 
        return X  + Z


    def tensor_biclustering(self):
        # create a random vector 
        v = np.random.rand(self._m)
        v = v/la.norm(v)   # norm of v should be equal to 1

        J1_true = list(range(0,  self._k1))
        J2_true = list(range(0,  self._k2))

        # Generating signal tensor
        X = np.zeros((self._m,self._n1,self._n2))
        for i in J1_true:
            for j in J2_true:
                X[:,i,j] = (self._sigma / np.sqrt(self._k1*self._k2)) * v
        

        Z = np.random.normal(0,1, (self._m, self._n1, self._n2)) # random normal standard distribution N(0,1) with size (m, n1,n2)
           
        T = X + Z
        return T
        