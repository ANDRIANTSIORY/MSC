import generate_3_D_tensor as gdata
import numpy as np
import fonctions as f
import multiSliceClustering as s
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt 
import scipy.linalg as la
from tensorly.decomposition import CP, Tucker
from sklearn.cluster import KMeans
import itertools


def comparaison_msc_Cpkmeans_Tuckerkmeans():
    dim = 50  # dimension of each element in each mode
    card = 10
    J1 = [i for i in range(card)]
    J2 = [i for i in range(card)]
    J3 = [i for i in range(card)]
    J = [J1, J2, J3]
    # rand index
    true1 = [0 for i in range(card)] + [1 for i in range(dim-card)]
    true = [true1, true1, true1]

    resultat, resultat_std, resultat_CP, resultat_Tucker =[],[],[],[]
    resultat_CPstd, resultat_Tuckerstd =[],[]
    mse_MSC, mse_CP, mse_TUCKER = [], [], []
    for sigma in range(20, 150, 2):
        # repetition 10 times for each value of sigma 
        rand_msi,rand_CP, rand_TUCKER = [],[],[]   # ARI
        mse_msc, mse_cp, mse_tucker = [], [], []   # we need the minimum for the sub-cube

        for i in range(10):
            #generate the data
            D = gdata.Generate_tensor(dim,dim,dim,k1=card,k2=card,k3=card, sigma=sigma) 
            D3 = D.rank_one()
            # Find the tricluster
            msc = s.MSC(D3, e_=0.0006)
            I = msc.get_cluster()

            # rand index
            estimation = [[1 if h in I[0] else 0 for h in range(50)],[1 if h in I[1] else 0 for h in range(50)],[1 if h in I[2] else 0 for h in range(50)]]
            rand_msi.append(f.find_adjusted_rand_score(true, estimation))

            # the MSE (mean square error of each sub-cube)
            mse_msC = []
            j0 = [[h for h in I[0]], [h for h in range(dim) if h not in I[0]]]
            j1 = [[h for h in I[1]], [h for h in range(dim) if h not in I[1]]]
            j2 = [[h for h in I[2]], [h for h in range(dim) if h not in I[2]]]
            for h in itertools.product(j0,j1,j2):
                cube = D3[h[0],:,:][:,h[1],:][:,:,h[2]]
                mse_msC.append(np.sum(( cube - cube.mean())**2) / (len(h[0])*len(h[1])*len(h[2])))

            mse_msc.append(np.min(mse_msC))


            # CP and Tucker  + k means
            factors = CP(rank=2)
            a = factors.fit_transform(D3) 
            rand_cp = []
            for g in range(3):  # for three factor
                kmeans = KMeans(n_clusters=2, random_state=0).fit(a[1][g])
                if g==0:
                    j0 = [[a for a, h in enumerate(kmeans.labels_) if h==0], [a for a, h in enumerate(kmeans.labels_) if h==1]]

                elif g==1:
                    j1 = [[a for a, h in enumerate(kmeans.labels_) if h==0],  [a for a, h in enumerate(kmeans.labels_) if h==1]]
                else:
                    j2 = [[a for a, h in enumerate(kmeans.labels_) if h==0], [a for a, h in enumerate(kmeans.labels_) if h==1]]

                rand_cp.append(adjusted_rand_score(true1, kmeans.labels_))

            rand_CP.append(np.mean(rand_cp))

            # the MSE (mean square error of each sub-cube)

            mse_cP = []

            for h in itertools.product(j0,j1,j2):
                cube = D3[h[0],:,:][:,h[1],:][:,:,h[2]]
                mse_cP.append(np.sum(( cube - cube.mean())**2) / (len(h[0])*len(h[1])*len(h[2])))

            mse_cp.append(np.min(mse_cP))
            #--------------------------------------------
            factors = Tucker(rank=2)
            a = factors.fit_transform(D3)
            rand_tucker = []

            for g in range(3):
                kmeans = KMeans(n_clusters=2, random_state=0).fit(a[1][g])
                if g==0:
                    j0 = [[a for a, h in enumerate(kmeans.labels_) if h==0], [a for a, h in enumerate(kmeans.labels_) if h==1]]
                elif g==1:
                    j1 = [[a for a, h in enumerate(kmeans.labels_) if h==0], [a for a, h in enumerate(kmeans.labels_) if h==1]]
                else:
                    j2 = [[a for a, h in enumerate(kmeans.labels_) if h==0], [a for a, h in enumerate(kmeans.labels_) if h==1]]

                rand_tucker.append(adjusted_rand_score(true1, kmeans.labels_))

            rand_TUCKER.append(np.mean(rand_tucker))

            # the MSE (mean square error of each sub-cube)
            mse_tuckeR = []
            for h in itertools.product(j0,j1,j2):
                cube = D3[h[0],:,:][:,h[1],:][:,:,h[2]]
                mse_tuckeR.append(np.sum(( cube - cube.mean())**2) / (len(h[0])*len(h[1])*len(h[2])))

            mse_tucker.append(np.min(mse_tuckeR))

        resultat.append((sigma, np.mean(rand_msi)))
        resultat_std.append((sigma, np.std(rand_msi)))
        resultat_CP.append(np.mean(rand_CP))
        resultat_Tucker.append(np.mean(rand_TUCKER))
        resultat_CPstd.append(np.std(rand_CP))
        resultat_Tuckerstd.append(np.std(rand_TUCKER))

    # mse of the sub-cube
    mse_MSC.append((np.min(mse_msc), np.std(mse_msc)))
    mse_CP.append((np.min(mse_cp), np.std(mse_cp)))
    mse_TUCKER.append((np.min(mse_tucker), np.std(mse_tucker)))

    # Plot the algorithm's quality
    sigma =  [resultat[i][0] for i in range(len(resultat))]

    ari = [resultat[i][1] for i in range(len(resultat))]
    std_rand = [resultat_std[i][1] for i in range(len(resultat_std))]

    plt.errorbar(sigma, ari, std_rand, linestyle='None', marker='')
    plt.plot(sigma,ari, label="msc")
    plt.errorbar(sigma, resultat_CP, resultat_CPstd, linestyle='None', marker='')
    plt.plot(sigma, resultat_CP, label="CP+k-means")
    plt.errorbar(sigma, resultat_Tucker, resultat_Tuckerstd, linestyle='None', marker='')
    plt.plot(sigma, resultat_Tucker, label="Tucker+k-means")
    plt.xlabel("Gamma")
    plt.ylabel("adjusted rand-index")
    plt.legend()
    plt.show()
    #plt.errorbar(sigma, ari, std_rand, linestyle='None', marker='')
    #plt.savefig('./test/recovery_rate_and_ari_triclustering0_0001.png')

if __name__ == "__main__":

    comparaison_msc_Cpkmeans_Tuckerkmeans()