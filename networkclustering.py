import pickle
from scipy.sparse import csc_matrix
import numpy as np
import scipy.sparse.linalg as spla
from random import sample, seed
from numpy.linalg import norm
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import normalize

'''
adj_file = 'adjacentMatrixUnderCS'
TODO: adj_file_pert: add edge between clusters
adj_file_pert =
'''  

def networkClustering(adj_file, data, number_of_topic = 10, sample_time = 10, edge_perturbation = False):
	seed(0)
	# create adjacency matrix and Laplacian matrix
	with open(adj_file, 'rb') as INFILE:
		adj_matrix = pickle.load(INFILE)

	node_num = adj_matrix.shape[0]
	degree = adj_matrix.sum(axis = 1).A1
	diag_index = np.arange(node_num)
	D = csc_matrix((degree, (diag_index, diag_index)), shape=(node_num, node_num))
	L = D - adj_matrix

	# spectral clustering
	val, U = spla.eigsh(L, k = number_of_topic)

	# row_norm = norm(U, axis = 1, ord = 2)
	U_norm = normalize(U, norm = 'l2', axis = 1)
	kmeans = KMeans(n_clusters = number_of_topic, random_state = 0).fit(U_norm)  
	label_pred = kmeans.labels_ 
	print Counter(label_pred)
	label_true = data._doc_label
	# if true label is more than one, randomly assign the document to one cluster, select the best results 
	NMI_list = np.zeros(sample_time)
	
	'''
	TODO: overlap cluster NMI
	'''
	for i in range(sample_time):
		label_true_unique = list()
		for item in label_true:
			if len(item) > 1:
				label_true_unique.append(sample(item, 1)[0])
			else:
				label_true_unique.append(list(item)[0])
		NMI_list[i] = normalized_mutual_info_score(label_true_unique, label_pred)
	NMI_best = max(NMI_list)
	print 'The NMI between spectral clustering and true label is\t' + str(NMI_best)
	return NMI_best
