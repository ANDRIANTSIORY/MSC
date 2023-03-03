from scipy.io import loadmat
import numpy as np
import fonctions as f
import multiSliceClustering as s
from tensorly.base import unfold
import itertools
import random
from numpy import linalg as la


def msc_realdata():

    annots = loadmat('../../data/Flow_Injection/fia.mat')

    d = annots['X']

    data = np.reshape(d, (12,100,89), order="F")
    print("data dimension :", data.shape)

    # MSC
    epsilon = 0.00013
    D = s.MSC(data, e_=epsilon)
    I = D.get_cluster()
    print("Cluster indices :", [I[0], I[1], I[2]])
    return ( [data, I])


# Evaluation
def evaluation( data, I):
    # frobenius norm of difference between slices
    # add one element randomly in I[i], i = 0,1,2 (mode)
    mode = 1         # choose the mode
    result_slice = []
    result_fiber_corr = []
    result_triclustering = []
    # we repeat several times the test
    for p in range(2):
        liste = [j for j in range(12) if j not in I[mode]]   # all indices
        new_I = I[mode].copy()
        new_I.append(random.choice(liste))
        norm_frob = []
        for k in new_I :
            for j in new_I:
                if (j > k):
                    if mode == 0:
                        norm_frob.append(la.norm(data[k,:,:] - data[j,:,:], 'fro'))
                    elif mode == 1:
                        norm_frob.append(la.norm(data[:,k,:] - data[:,j,:], 'fro'))
                    else:
                        norm_frob.append(la.norm(data[:,:,k] - data[:,:,j], 'fro'))
        result_slice.append(max(norm_frob))
        if mode == 0:
            result_fiber_corr.append(f.fiber_correlation(data, new_I, I[2], (mode+1)%3))
            result_triclustering.append( f.rmse(data,new_I, I[1], I[2]))
        elif mode == 1:
            result_fiber_corr.append(f.fiber_correlation(data, I[0], new_I, (mode+1)%3))
            result_triclustering.append(f.rmse(data,I[0], new_I, I[2]))
        else:
            result_fiber_corr.append(f.fiber_correlation(data, I[1], new_I, (mode+1)%3))
            result_triclustering.append( f.rmse(data, I[0], I[1], new_I))
    #print("min = ", min(norm_frob), " and max = ", max(norm_frob))
    print("min Slice :", np.min(result_slice))   #np.min(norm_frob))
    print("min correlation fiber mode ",(mode+1)%3 ," = ", np.min(result_fiber_corr))
    print("min mse tricluster :", np.min(result_triclustering))


if __name__ == "__main__":
    a = msc_realdata()
    print("--------------- EVALUATION -----------------")
    evaluation(a[0], a[1])

