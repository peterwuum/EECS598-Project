import pickle
from scipy.sparse import csc_matrix
import numpy as np
import scipy.sparse.linalg as spla
import random
from numpy.linalg import norm
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import normalize


#adj_file = 'adjacentMatrixUnderCS'
#adj_file_pert = 
def networkClustering(adj_file, number_of_topic = 10, sample_time = 10, edge_perturbation = False)
	# create adjacency matrix and Laplacian matrix
    adj_file = open(adj_file,'rb')
    adj_matrix = pickle.load(adj_file)
	node_num = adj_matrix.shape[0]
	degree = adj_matrix.sum(axis = 1).A1
	diag_index = np.arange(node_num)
	D = csc_matrix((degree, (diag_index, diag_index)), shape=(node_num, node_num))
	L = D - adj_matrix
	K = number_of_topic

	# spectral clustering
	val,U = spla.eigsh(L, k = K)
	row_norm = norm(U, axis = 1, ord = 2)
	U_norm = normalize(U, norm = 'l1',axis = 1)
	kmeans = KMeans(n_clusters=K, random_state=0).fit(U)  
	label_pred = kmeans.labels_ 
	label_true = data._doc_label
	# if true label is more than one, randomly assign the document to one cluster, select the best results 
	NMI_list = np.zeros(sample_time)
	for i in range(sample_time)
		label_true_unique = [random.sample(temp,1)[0] for temp in label_true if len(temp) > 1]
    	NMI_list[i] = normalized_mutual_info_scorel(label_true_unique, label_pred)
    NMI_best = max(NMI_list)
    print 'The NMI between spectral clustering and true label is' + str(NMI_best)
    return NMI_best
