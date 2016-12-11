'''
netplsa model using multi processing method to speed up and accept more data
using sparse matrix to store the data
'''

import numpy as np
from scipy.sparse import csr_matrix
import re
from collections import Counter

row = list()
col = list()
data = list()
wordlist = list()
doc_path = 'titlesUnderCS_10000.txt'

count = 0
for i in range(100):
	with open(doc_path, 'r') as INFILE:
		for line in INFILE.readlines():
			temp = Counter(re.split(' ', line.strip()))
			for key, val in temp.items():
				if key not in wordlist:
					wordlist.append(key)
				row.append(count)
				col.append(wordlist.index(key))
				data.append(val)
			count += 1
print 'data size: ' + str(len(data))
print 'word size: ' + str(len(wordlist))
print 'number of document: ' + str(count)

doc_term_matrix = csr_matrix((data, (row, col)), shape = (count, len(wordlist)))
print doc_term_matrix.toarray()



# todo: parameter passing into the EStep and MStep are the start and end
