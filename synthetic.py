import numpy as np
from scipy import sparse
import pickle

N = 1000
a = np.zeros((N, N))
for i in range(0, N):
	for j in range(i + 1, N):
		a[i][j] =  np.random.rand(1)
		if a[i][j] < 0.7:
			a[i][j] = 0
		else:
			a[i][j] = 1


a_symm = a + a.T
print a_symm

sA = sparse.csr_matrix(a_symm)

print sA[0 : 100]

with open('synthetic_data', 'wb') as INFILE:
	pickle.dump(sA, INFILE)