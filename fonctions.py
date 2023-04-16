from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
from tensorly.base import unfold

# remove liste of slice in fixed dimension
def rem(donnee, dim, liste):
    while (liste != []):
        dimT = list(donnee.shape)
        if dim ==1:
            dimT[1] = dimT[1] - 1
            
            donneeI = np.zeros(dimT)
            for i in range(dimT[1]):
                if (i >= liste[-1]):
                    donneeI[:,1,:] = donnee[:,i+1,:]
                else:
                    donneeI[:,i,:] = donnee[:,i,:]
            donnee = donneeI  
            liste = liste[:-1]
        elif dim == 2:
            dimT[2] = dimT[2] - 1
            
            donneeI = np.zeros((dimT))
            for i in range(dimT[2]):
                if (i >= liste[-1]):
                    donneeI[:,:,i] = donnee[:,:,i+1]
                else:
                    donneeI[:,:,i] = donnee[:,:,i]
            donnee = donneeI  
            liste = liste[:-1]
            
        elif dim == 0:
            dimT[0] = dimT[0] - 1
            
            donneeI = np.zeros((dimT))
            for i in range(dimT[0]):
                if (i >= liste[-1]):
                    donneeI[i,:,:] = donnee[i+1,:,:]
                else:
                    donneeI[i,:,:] = donnee[i,:,:]
            donnee = donneeI  
            liste = liste[:-1]
                
    return donnee


# recovery rate
# I is the list the true cluster
# J is the list of estimated cluster
def recovery_rate(I, J):
    r_rate = 0
    for i in range(3):
        r  = set(I[i]).intersection(set(J[i]))
        r_rate += (len(r) / (3 * len(J[0])))
    return r_rate




def gound_truth_known_tensor_biclustering(true, estimation):  # true is a couple and the estimation as well
    # recovery rate
    r = len(set(true[1]).intersection(set(estimation[1]))) / (2*len(true[1]))
    r += len(set(true[0]).intersection(set(estimation[0]))) / (2*len(true[0]))
    #rint("recovery rate : ", r)
    return r


def find_adjusted_rand_score(vrai, estimation):
    result = 0
    for i in range(len(vrai)):
        result += adjusted_rand_score(vrai[i], estimation[i])
    return result/len(vrai)

# TOp eigenvector  and eigenvalue of square matrix 
# Rayleigh quotient method

def rayleigh_power_iteration(A, num_simulations: int, eps = 0.000001):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    a = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, a)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

        # test the convergence
        if np.linalg.norm(b_k - a) <= eps :
            break
        else :
            a = b_k

    mu = np.dot(np.dot(b_k.T, A),b_k)

    return  mu, b_k

# correlation of fiber
def fiber_correlation(data,a, b, mode):
        if mode == 0:
            return np.mean(np.abs(np.corrcoef(unfold(data[:,a,:][:,:,b],mode).T)))
        elif mode ==1:
            return np.mean(np.abs(np.corrcoef(unfold(data[a,:,:][:,:,b],mode).T)))
        else:
            return np.mean(np.abs(np.corrcoef(unfold(data[a,:,:][:,b,:],mode).T)))
def rmse(data,a, b, c):
    resultat = data[a,:,:][:,b,:][:,:,c]
    return (np.sum((resultat - np.mean(resultat))*(resultat - np.mean(resultat))))**0.5
