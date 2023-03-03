import numpy as np
from sklearn import preprocessing
import scipy.linalg as la
import heapq
import seaborn as sn
from sklearn.preprocessing import normalize


class Tensorbiclustering():
    def __init__(self, tensor, block = 1, norm = ""):
        self.tensor = tensor
        self.block = block
        self.norm = norm
        self.dim = self.tensor.shape
        self.c1c2()

    def c1c2(self):
        
        self.C2 = np.zeros((self.dim[2],self.dim[2]))
        self.C1 = np.zeros((self.dim[1],self.dim[1]))
        if (self.norm == ""):
            for i in range(self.dim[1]):
                self.C2 += (self.tensor[:,i,:].transpose()).dot(self.tensor[:,i,:])
            # lateral slice, find J_1
            for i in range(self.dim[2]):
                self.C1 += (self.tensor[:,:,i].transpose()).dot(self.tensor[:,:,i])
        elif (self.norm == "divbymax"):
            for i in range(self.dim[1]):
                data = self.tensor[:,i,:]/(self.tensor[:,i,:].max(axis=0))
                self.C2 += (data.transpose()).dot(data)
            # lateral slice, find J_1
            for i in range(self.dim[2]):
                data = self.tensor[:,:,i]/self.tensor[:,:,i].max(axis=0)
                self.C1 += (data.transpose()).dot(data)
        elif (self.norm =="centered"):
            # Numpy broadcasting
            for i in range(self.dim[1]):
                col_mean = np.mean(self.tensor[:,i,:], axis = 0)
                st = self.tensor[:,i,:] - col_mean
                self.C2 += (st.transpose()).dot(st)
            # lateral slice, find J_1
            for i in range(self.dim[2]):
                col_mean = np.mean(self.tensor[:,:,i], axis = 0)
                st = self.tensor[:,:,i] - col_mean
                self.C1 += (st.transpose()).dot(st)
        elif (self.norm == "normalize"):
            for i in range(self.dim[1]):
                norme = normalize(self.tensor[:,i,:], axis = 0)
                self.C2 += (norme.transpose()).dot(norme)
            for i in range(self.dim[2]):
                norme = normalize(self.tensor[:,:,i], axis = 0)
                self.C1 += (norme.transpose()).dot(norme)

        self.weight_C1, self.vector_C1 = la.eig(self.C1)
        self.weight_C1, self.vector_C1 = self.weight_C1[:self.block].real, np.abs(self.vector_C1[:,:self.block])
      
        self.weight_C2, self.vector_C2 = la.eig(self.C2)
        self.weight_C2, self.vector_C2 = self.weight_C2[:self.block].real, np.abs(self.vector_C2[:,:self.block])
        
    
      
                
  