import generate_3_D_tensor as gdata
import numpy as np
import fonctions as f
import multiSliceClustering as s
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt 
import Tensor_Foldin_Spectral as tfs
import scipy.linalg as la
import heapq
import itertools
from tensorly.base import unfold



def msc_tfs():
    #generate the data
    result, result1 = [], []
    for sigma in range(20, 200,5):
        rate_msc, corr_msc, corr_tfs, rate_tfs = [], [], [], []
        n = 70
        m = 50
        l = 10
        true_ari = [[0 if i>l else 1 for i in range(n)],[0 if i>l else 1 for i in range(n)]]
        for i in range(2):
            D = gdata.Generate_tensor(m,n,n,k1=l,k2=l, sigma=sigma) 
            D1 = D.tensor_biclustering() 
            #D1 = abs(D1)  
   
            msc = s.MSC(D1, e_=0.00027)
            I = msc.get_cluster()

            estimation = [I[1], I[2]]
            true = [[i for i in range(l)], [i for i in range(l)]]
            rate_msc.append(f.gound_truth_known_tensor_biclustering(true, estimation))
            # correlation of trajectories
            trajectories = unfold(D1[:,:,I[2]][:,I[1],:],0)
            corr_msc.append(np.mean(np.abs(np.corrcoef(trajectories.T))))
        
            # Method tensor folding and spectral
            T = tfs.Tensorbiclustering(D1, block=1)
            J1 = sorted(range(len(T.vector_C1)), key = lambda sub: T.vector_C1[sub])[-l:] 
            J2 = sorted(range(len(T.vector_C2)), key = lambda sub: T.vector_C2[sub])[-l:] 
            estimation = [J1, J2]
            rate_tfs.append(f.gound_truth_known_tensor_biclustering(true, estimation))
            # the correlation of the trajectories
            trajectories = unfold(D1[:,:,J2][:,J1,:],0)
            corr_tfs.append(np.mean(np.abs(np.corrcoef(trajectories.T))))
        
        
        result.append((sigma, np.mean(rate_msc),np.mean(rate_tfs),np.std(rate_msc),np.std(rate_tfs)))   # recovery rate
        result1.append((sigma, np.mean(corr_msc),np.mean(corr_tfs),np.std(corr_msc),np.std(corr_tfs)))  # correlation


    # ------------------------------------------------------------------
    sigma =  [result[i][0] for i in range(len(result))]
    # Multi-scile
    rate_msc=[result[i][1] for i in range(len(result))]
    rate_tfs=[result[i][2] for i in range(len(result))]
    rate_msc_std = [result[i][3] for i in range(len(result))]
    rate_tfs_std = [result[i][4] for i in range(len(result))]
    # correlation
    sigma =  [result1[i][0] for i in range(len(result1))]
    corr_msc =[result1[i][1] for i in range(len(result1))]
    corr_tfs =[result1[i][2] for i in range(len(result1))]
    corr_msc_std = [result1[i][3] for i in range(len(result1))]
    corr_tfs_std = [result1[i][4] for i in range(len(result1))]

    #plt.errorbar(sigma, rate_msc, rate_msc_std, linestyle='None', marker='')
    plt.plot(sigma,rate_msc, label="rec - MSC")
    #plt.errorbar(sigma, rate_tfs, rate_tfs_std, linestyle='None', marker='')
    plt.plot(sigma,rate_tfs, label="rec - TFS")
    plt.legend(loc='upper left')
    plt.xlabel("Gamma", fontsize = 18)
    plt.ylabel("Recovery rate", fontsize=18)
    plt.legend(loc='lower right', fontsize = 15)
    #plt.savefig('./image/tfs_msc_rate.png')
    plt.show()
    #plt.errorbar(sigma, corr_msc, corr_msc_std, linestyle='None', marker='', color="cyan")
    plt.plot(sigma,corr_msc, label="corr - MSC")
    #plt.errorbar(sigma, corr_tfs, corr_tfs_std, linestyle='None', marker='', color='gray')
    plt.plot(sigma,corr_tfs, label="corr - TFS")
    plt.legend(loc='lower right', fontsize=15)
    plt.xlabel("Gamma", fontsize=18)
    plt.ylabel("trajectories' correlation", fontsize=18)
    #plt.savefig('./image/tfs_msc_correlation.png')
    plt.show()

if __name__=="__main__":
    msc_tfs()
