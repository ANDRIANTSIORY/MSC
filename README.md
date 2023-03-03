# Multi-Slice clustering (MSC) for 3-order tensor

The MSC algorithm is a clustering algorithm applied to a third order tensor $\mathcal{T}\in\mathbb{R}^{m_1\times m_2\times m_3}$, $\mathcal{T} = \mathcal{X}+\mathcal{Z}$. 

* $\mathcal{X}$ is the signal tensor
* $\mathcal{Z}$ is the noise tensor

The theoretical approach assumes that the signal tensor  is a rank one tensor. 

![an image](signal_rank_one.png)

For each mode of $\mathcal{T}$, this method studies each matrix slice. We select the top-eigenvectors of the covariance matrix of all slices and study a covariance-type matrix associated with those. This guides us towards the selection of an index subset  with highly similar features.  

**The similarity within the group is controlled by the parameter given to the algorithm.**

![an image](msc.png)

**Theorem** : Let $|J_1|=l$, assume that $\sqrt{\epsilon}\le \frac{1}{m_1-l}$. $\forall i,n\in J_1 $, for $\lambda = \mathcal{O}(\mu)$ we have
		$$
				|d_i - d_{n}| \le l\frac{\epsilon}{2} + \sqrt{\log(m_1-l)}
		$$
			with probability at least $1-e(m_1-l)^{-c_1}$ with a constant $c_1>0$.


where 		$d_i =  \sum_{j\in [m_1]} c_{i,j} = \sum_{j\in [m_1]} | \langle \tilde\lambda_i \mathbf{v}_i,\tilde\lambda_j \mathbf{v}_j \rangle| . $


[1] Andriantsiory, D.F., Ben Geloun, J., Lebbah, M.: Multi-slice clustering for 3-order tensor data - supplementary material. arXiv preprint arXiv:2109.10803 (2021)

[2] Andriantsiory, D.F., Geloun, J.B., Lebbah, M.: Multi-slice clustering for 3-order tensor. In: 2021 20th IEEE International Conference on Machine Learning and Applications (ICMLA), pp. 173–178 (2021)
