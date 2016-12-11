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
	label_pred = np.array(kmeans.labels_)
	label_true = data._doc_label

	pred_label_matrix = np.zeros((len(label_pred), number_of_topic), dtype = 'f')
	index_row = np.arange(len(label_pred))
	pred_label_matrix[index_row, label_pred] = 1


	true_label_matrix = np.zeros((len(label_true), number_of_topic),dtype = 'f')
	for index in range(0,len(label_true)):
		for i in label_true[index]:
			true_label_matrix[index][i] = 1

	NMI = overlapNMI(true_label_matrix, pred_label_matrix)

	print 'The NMI between spectral clustering and true label is\t' + str(NMI)
	return NMI