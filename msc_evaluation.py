import generate_3_D_tensor as gdata
import numpy as np
import fonctions as f
import multiSliceClustering as s
import matplotlib.pyplot as plt 
import scipy.linalg as la
from tensorly.decomposition import CP, Tucker
from sklearn.metrics.cluster import adjusted_rand_score

def first_experiment():
	dim = 50  # dimension of each element in each mode
	card = 10  # cardinal of J1
	J1 = [i for i in range(card)]
	J2 = [i for i in range(card)]
	J3 = [i for i in range(card)]
	J = [J1, J2, J3]
	# rand index
	true = [0 for i in range(card)] + [1 for i in range(dim - card)]
	true = [true, true, true]

	resultat, resultat_std, S, S_std = [], [], [], []
	for sigma in range(20, 150, 2):  # range(20, 150, 2)
		# repetition 10 times for each value of sigma 
		rate, sim = [], []   # recovery rate
		rand = [0]   # ARI not used, this is just a value to avoid error

		for i in range(10):  #for i in range(10)

			#generate the data
			D = gdata.Generate_tensor(dim,dim,dim,k1=card,k2=card,k3=card, sigma=sigma) 
			D3 = D.rank_one()
			# MSC method
			msc = s.MSC(D3, e_=0.0006, method="") # method="rayleigh"
			I = msc.get_cluster()

			# recovery rate
			rate.append( f.recovery_rate(I, J) )
			sim.append(np.mean(msc.get_similarity()))

    		# rand index
    		#estimation = [[1 if i in I[0] else 0 for i in range(50)],[1 if i in I[1] else 0 for i in range(50)],[1 if i in I[2] else 0 for i in range(50)]]
    		#rand.append(f.find_adjusted_rand_score(true, estimation))

		resultat.append((sigma, np.mean(rate), np.mean(rand)))
		resultat_std.append((sigma, np.std(rate), np.std(rand)))
		# result[:][0]: the variation of sigma
		# result[:][1]: recovery rate
		# result[:][2]: rand index
		S.append(np.mean(sim))
		S_std.append(np.std(sim))




    # --------------- plot ------------------------------------   

	# Plot the algorithm's quality
	sigma =  [resultat[i][0] for i in range(len(resultat))]
	rate = [resultat[i][1] for i in range(len(resultat))]
	#ari = [resultat[i][2] for i in range(len(resultat))]
	std_rate = [resultat_std[i][1] for i in range(len(resultat_std))]
	#std_rand = [resultat_std[i][2] for i in range(len(resultat_std))]
	plt.errorbar(sigma, rate, std_rate, linestyle='None', marker='')
	plt.plot(sigma, S, label="Similarity")
	plt.errorbar(sigma, S, S_std, linestyle='None', marker='')
	plt.plot(sigma,rate, label="Recovery rate")
	plt.xlabel("gamma")
	plt.ylabel("Recovery rate")
	#plt.hlines(0.7, 20, 150,color = "gray", linestyles="dotted")
	plt.hlines(0.9, 20, 150,color = "gray", linestyles="dotted")
	plt.show()
	#plt.errorbar(sigma, ari, std_rand, linestyle='None', marker='')
	#plt.savefig('./test/recovery_rate_and_ari_triclustering0_0001.png')


if __name__ == "__main__":

	first_experiment()


