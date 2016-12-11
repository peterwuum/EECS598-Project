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
from overlapNMI import overlapNMI

'''
adj_file = 'adjacentMatrixUnderCS'
TODO: adj_file_pert: add edge between clusters
adj_file_pert =
'''  

def networkClustering(adj_file, data, number_of_topic = 10, edge_perturbation = False):
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
	label_pred = np.arange(kmeans.labels_)
	print Counter(label_pred)
	label_true = data._doc_label
	# if true label is more than one, randomly assign the document to one cluster, select the best results 
	#NMI_list = np.zeros(sample_time)
	
	'''
	TODO: overlap cluster NMI
	'''

	pred_label_matrix = np.zeros(len(label_pred), number_of_topoc),dtype = 'f')
	index_row = np.arange(len(label_pred))
	pred_label_matrix[index_row, label_pred] = 1

	true_label_matrix = np.zeros(len(label_true), number_of_topoc),dtype = 'f')
	for index in range(0,len(label_true)):
		for i in label_true[index]:
			true_label_matrix[index][i] = 1

	NMI = overlapNMI(true_label_matrix, pred_label_matrix)

	# for i in range(sample_time):
	# 	label_true_unique = list()
	# 	for item in label_true:
	# 		if len(item) > 1:
	# 			label_true_unique.append(sample(item, 1)[0])
	# 		else:
	# 			label_true_unique.append(list(item)[0])
	# 	NMI_list[i] = normalized_mutual_info_score(label_true_unique, label_pred)
	# NMI_best = max(NMI_list)
	print 'The NMI between spectral clustering and true label is\t' + str(NMI)
	return NMI_best
