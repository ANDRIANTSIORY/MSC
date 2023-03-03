from sklearn import preprocessing
import scipy.linalg as la
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.decomposition import PCA
import fonctions as f


class MSC():
    def __init__(self, tensor, norm="centralized", e_= 0.0, method=""):
        self._tensor = tensor
        self._norm = norm
        self._e_ = e_
        self._method = method
        self._dim = self._tensor.shape
        self._cluster_, self._similarity_ = self.tricluster_method()



    def normed_and_covariance(self, M):
        if self._norm == "centralized":
            m = np.mean(M, axis=0).reshape(1,-1)  # mean of each column
            M = M - m
            M = (1/len(M[0,:])) * ((M.T).dot(M))
        elif self._norm == "normalized":
            M = normalize(M, axis=0)
            M = (M.T).dot(M)
        return M


    def tricluster_method(self):
        l = len(self._dim)   # l = 3
        # the eigenvalue and eigenvector of each slice for each dimension
        for i in range(l):
            if i == 0:
                e0 = []
                for k in range(self._dim[0]):
                    frontal =  self.normed_and_covariance(self._tensor[k,:,:])
                    if (self._method == "rayleigh"):
                        w, v = f.rayleigh_power_iteration(frontal, 20)
                        e0.append([w.real, v.real])
                    else :
                        w, v = la.eig(frontal)
                        e0.append([w[0].real, v[:,0].real])
                    
            elif i == 1:
                e1 = []
                for k in range(self._dim[1]):
                    horizontale = self.normed_and_covariance(self._tensor[:,k,:])
                    if (self._method == "rayleigh"):
                        w, v = f.rayleigh_power_iteration(horizontale, 10)
                        e1.append([w.real, v.real])
                    else :
                        w, v = la.eig(horizontale)
                        e1.append([w[0].real, v[:,0].real])
                    
                    
            elif i==2:
                e2 = []
                for k in range(self._dim[2]):
                    laterale = self.normed_and_covariance(self._tensor[:,:,k])
                    if (self._method == "rayleigh"):
                        w, v = f.rayleigh_power_iteration(laterale, 10)
                        e2.append([w.real, v.real])
                    else :
                        w, v = la.eig(laterale)
                        e2.append([w[0].real, v[:,0].real])


        e = [e0, e1, e2]

        j = 0

        result = []
        weight = []
        similarity = []
        self._d = []
        counteur = 0  # conteur
        for i in e:
            lam = [k[0] for k in i]   # The eigenvalues
            weight.append(lam)
            lam = np.max(lam)
            
            M = np.zeros((len(i[0][1]), len(i)))  # 

            for k in range(len(i)):
                M[:,k] = (i[k][0] * i[k][1] ) / lam  # product of eigenvalue and eigenvector of slice k
                

            M = (M.T).dot(M)     # covariance matrix  - symmetric matrix
            Matrices = np.abs( M.copy() )
            
            M = np.sum(np.abs(M), axis = 0)  # sum of each line of M
            
            M_order = M.copy()
            self._d.append(M)
            M_order.sort()
            
            gap = self.initialize_c(M_order) 
            
            thm1 = [k for k, l in enumerate(M) if l >= gap]   # M is the original vector
            thm = thm1.copy()
            value  = [M[k] for k in thm1]    # The value for all indices in thm1 
            c = len(thm1)   

            # verification of theorem 2
            # c_prime = self.find_cprime(self._e_, len(M) - c)
            #if ( self._e_**0.5 <= c_prime / (len(M)-c) and c_prime <= np.sqrt((M) - c)) :
            #    while ((self.max_difference(value) > (c * (self._e_ / 2)) + (len(M) - c)**0.5) ) :
            #        thm = [k for k, l in enumerate(M) if l > min(value)]
            #        value.remove(min(value))
            #        c -= 1
            #        if (value == list()):
            #            thm = thm1
            #            break
            #        c_prime = self.find_cprime(self._e_, len(M) - c)

            # Verification of theorem 1

            if (self._e_ == 0.0 ) :

                self._e_ = 1 / ((len(M) - c)**2)

                #if ( self._e_**0.5 <= 1/(len(M)-c)) :
                while ((self.max_difference(value) > (c * (self._e_ / 2)) + (np.log(len(M)-c)**0.5)) ):
                    thm = [k for k, l in enumerate(M) if l > min(value)]
                    value.remove(min(value))
                    c -= 1
                    if (value == list()):
                        thm = thm1
                        break
                    self._e_ = 1 / ((len(M) - c)**2)   # update the value of epsilon
                print ("epsilon - ",counteur," = ", self._e_)
                self._e_ = 0.0

            elif (self._e_ < 1 / ((len(M) - c)**2)) :   # with a fix epsilon
                while ((self.max_difference(value) > (c * (self._e_ / 2)) + (np.log(len(M)-c)**0.5)) ):
                    thm = [k for k, l in enumerate(M) if l > min(value)]
                    value.remove(min(value))
                    c -= 1
                    if (value == list()):
                        thm = thm1
                        break
            counteur += 1

            # mean of the similarity
            similarity.append(np.mean(Matrices[thm,:][:,thm]))

            # result is a list with three elements and its elements have type list
            result.append(thm)       

        return result,  similarity

    def find_cprime(self, a, b):
        if ((a**0.5) * b < np.sqrt(b )) :
            return int((a**0.5) * b)
        else:
            return b + 1

    def get_cluster(self):
        return self._cluster_
    def get_similarity(self):     # V^tV
        return self._similarity_
    def get_d(self):
        return self._d


    def initialize_c(self, Liste):   # Listte is the vector d in assending order
        i = 1
        c = abs(Liste[0] - Liste[1])
        for j in range(1, len(Liste)-1):
            if c < abs(Liste[j] - Liste[j+1]) :
                c = abs(Liste[j] - Liste[j+1])
                i = j

        return Liste[i]


    def max_difference(self, valeur):
        c = []
        for i in range(len(valeur)):
            for j in range(i, len(valeur)):
                c.append(np.abs(valeur[i]-valeur[j]))
        return max(c)



        

