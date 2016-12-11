import time
from collections import Counter, defaultdict
import re
import numpy as np
from scipy.sparse.csgraph import laplacian
from sklearn.preprocessing import normalize

from scipy.sparse import csr_matrix, lil_matrix
import pickle
from sklearn.feature_extraction.text import TfidfTransformer

from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
import click
import random


'''
TODO: change the EM to distributed version
'''

class PLSA(object):
	def __init__(self, doc_path, stop_word_path, path_to_adj, path_to_idname, path_to_paperid, 
					number_of_topic = 20, maxIteration = 30, threshold = 0.02, network = False, 
					lambda_par = 0.5, gamma_par = 0.1, synthetic_edge_prob = 0.0, 
					lemmatize = True, stemmer = False, 
					save = None, optimal = False):
		self._save = save
		# optimal means using the true label as doc_topic matrix initialization
		self._optimal = optimal
		self._doc_path = doc_path
		self._stopword = set()
		with open(stop_word_path, 'r') as INFILE:
			for line in INFILE.readlines():
				self._stopword.add(line.strip())
		
		self.number_of_topic = number_of_topic
		self._maxIteration = maxIteration
		self._threshold = threshold
		self._CommonWordList = list()

		self._numSelectedWord = 300
		self.doc_term_matrix = 0
		self._doc_topic = 0
		self._topic_word = 0
		self._probability = 0
		self._numDoc = 0
		self._numWord = 0

		self._wordnet_lemmatizer = False
		if lemmatize:
			self._wordnet_lemmatizer = WordNetLemmatizer()
		
		self._lancaster_stemmer = False
		if stemmer:
			self._lancaster_stemmer = LancasterStemmer()

		# read kid - keyword map
		self._label_category = dict()
		name2id = 0
		with open(path_to_idname, 'r') as INFILE:
			for line in INFILE.readlines():
				self._label_category[re.split('\t', line)[0]] = name2id
				name2id += 1

		# doc_label is a list of set of documents label ids
		self._doc_label = list()
		with open(path_to_paperid, 'r') as INFILE:
			for line in INFILE.readlines():
				ids = re.split('\t', line.strip())
				temp = set()
				for item in ids:
					temp.add(self._label_category[item])
				self._doc_label.append(temp)

		with open(path_to_adj, 'rb') as INFILE:
			self._adj = pickle.load(INFILE)
			self._adj = lil_matrix(self._adj)
		
		# Add synthetic edges
		if synthetic_edge_prob > 0:
			edges = []

			for i in range(0, self._adj.shape[0]):
				for j in range(i, self._adj.shape[0]):
					if i == j:
						continue

					keywords_set1 = self._doc_label[i]
					keywords_set2 = self._doc_label[j]
					intersection_size = len(keywords_set1 & keywords_set2)
					
					# If two paper are not belong to the same area
					if intersection_size == 0:
						edges.append((i, j))

			for i, j in edges:
				if random.uniform(0, 1) < synthetic_edge_prob:
					self._adj[i, j] = 1.0
					self._adj[j, i] = 1.0

		self._lap = laplacian(self._adj)
		self._adj = normalize(self._adj, norm='l1', axis=1)
		self.synthetic_edge_prob = synthetic_edge_prob
		self.network = network
		self._lambda = lambda_par
		self._gamma = gamma_par
		self._avg_iteration_time = []

		self._old = 1
		self._new = 1

	def _preprocessing(self):
		list_of_doc_word_count = list()
		# document frequency
		word_doc_list = defaultdict(int)
		
		with open (self._doc_path, 'r') as INFILE:
			for line in INFILE.readlines():
				temp = Counter(re.split(' ', line.strip()))
				
				temp = temp.items()
				temp_list = list()
				
				for item in temp:
					key = item[0]
					val = item[1]

					if self._wordnet_lemmatizer:
						try:
							word = str(self._wordnet_lemmatizer.lemmatize(str(self._wordnet_lemmatizer.lemmatize(key)), pos = 'v'))

						except:
							word = key

					elif self._lancaster_stemmer:
						try:
							word = str(self._lancaster_stemmer.stem(key))
						except:
							word = key

					else:
						word = key
					
					if word in self._stopword:
						temp.pop(temp.index(item))
						continue
					
					elif len(word) < 3:
						temp.pop(temp.index(item))
						continue
					
					else:
						word_doc_list[word] += 1
						if word not in self._CommonWordList:
							self._CommonWordList.append(word)
						if self._wordnet_lemmatizer or self._lancaster_stemmer:
							temp_item = list(item)
							temp_item[0] = word
							temp_list.append(tuple(temp_item))
				
				if self._wordnet_lemmatizer or self._lancaster_stemmer:
					list_of_doc_word_count.append(temp_list)
				else:
					list_of_doc_word_count.append(temp)
	

		self._numDoc = len(list_of_doc_word_count)
		print 'document'
		print self._numDoc
		# Check the number of document
		if self._numDoc != self._adj.shape[0]:
			raise Exception()
		"""
		count = 0
		temp = sorted(word_doc_list.items(), key = lambda x : x[1], reverse = True)
		
		for item in temp:
			print item[0] + '\t' + str(item[1])
			# if val >= 10:
			# 	print key
			# 	count += 1
		print '\n\n'

		min_threhold = 0.005 * self._numDoc
		max_threhold = 0.05 * self._numDoc

		for key, val in word_doc_list.items():
			if val < min_threhold or val > max_threhold:
				self._CommonWordList.pop(self._CommonWordList.index(key))
		"""

		self._numWord = len(self._CommonWordList)
		print 'word'
		print self._numWord
		
		# Create initial adjacent matrix
		self.doc_term_matrix = np.zeros(shape = (self._numDoc, self._numWord))
		
		for i in range(0, len(list_of_doc_word_count)):
			for item in list_of_doc_word_count[i]:
				if item[0] in self._CommonWordList:
					self.doc_term_matrix[i][self._CommonWordList.index(item[0])] += item[1]
		

		# Select words which have top k highest entropy
		col_sum = self.doc_term_matrix.sum(axis=0)
		p_matrix = self.doc_term_matrix / col_sum[np.newaxis, :]


		word_entropy = np.zeros(shape = self._numWord)

		for i in range(0, len(self._CommonWordList)):
			temp = p_matrix[:, i][p_matrix[:, i] > 0]
			word_entropy[i] = np.dot(temp, np.log(temp))

		print 'word_entropy size\t%s' % word_entropy.shape
		ind = np.argsort(word_entropy)
		_CommonWordListTmp = []
		_indexTmp = []
		for i in ind[::-1]:
			if word_entropy[i] < -2 and len(_CommonWordListTmp) < self._numSelectedWord:
				_CommonWordListTmp.append(self._CommonWordList[i])
				_indexTmp.append(i)

		self._CommonWordList = _CommonWordListTmp
		_CommonWordListSet = set(self._CommonWordList)

		self._numSelectedWord = len(self._CommonWordList)
		print ''
		print 'Selected words: '
		for i in range(self._numSelectedWord):
			print self._CommonWordList[i], '\t\t\t\t', word_entropy[_indexTmp[i]]
		print ''

		self._numWord = self._numSelectedWord
		self.doc_term_matrix = np.zeros(shape = (self._numDoc, self._numWord))

		for i in range(0, len(list_of_doc_word_count)):
			for item in list_of_doc_word_count[i]:
				if item[0] in _CommonWordListSet:
					self.doc_term_matrix[i][self._CommonWordList.index(item[0])] += item[1]		

		print 'Built processed doc term matrix with size (%d, %d)' % self.doc_term_matrix.shape

		print 'tfidf version'
		transformer = TfidfTransformer(smooth_idf = False)
		self.doc_term_matrix = transformer.fit_transform(self.doc_term_matrix).toarray()
		# print self.doc_term_matrix

	def _initParameters(self):
		normalization = np.sum(self._doc_topic, axis = 1)
		for i in range(0, len(normalization)):
			self._doc_topic[i] /= normalization[i]

		normalization = np.sum(self._topic_word, axis = 1)
		for i in range(0, len(normalization)):
			self._topic_word[i] /= normalization[i]

		self._avg_iteration_time.append(0)

	
	def _EStep(self):
		for k in range(0, self.number_of_topic):
			self._probability[:, :, k] = np.outer(self._topic_word[k, :], self._doc_topic[:, k]).T
		denominator = self._probability.sum(2)

		for k in range(0, self.number_of_topic):
			# if denominator == 0:
			self._probability[:, :, k][denominator == 0] = np.zeros((self._numDoc, self._numWord))[denominator == 0]	
			self._probability[:, :, k][denominator > 0] /= denominator[denominator > 0]

	def _MStep(self):
		old_loglikelihood = self._LogLikelihood()

		# update topic-word matrix
		for k in range(0, self.number_of_topic):
		   self._topic_word[k, :] = (self.doc_term_matrix * self._probability[:, :, k]).sum(axis = 0)
		#self._topic_word = normalize(self._topic_word, axis = 1, norm = 'l1')
		denominator = np.sum(self._topic_word, axis = 1)
		temp = np.zeros(self.number_of_topic)
		temp[denominator != 0] = np.ones(sum(denominator!=0))/denominator[denominator!=0]
		temp[denominator == 0] = np.ones(sum(denominator == 0))*1.0/self._numWord
		self._topic_word[denominator==0, :] = np.ones((sum(denominator == 0), self._numWord)) 
		self._topic_word = np.dot(np.diag(temp), self._topic_word)
		
		# update document-topic matrix
		for k in range(0,self.number_of_topic):
		   self._doc_topic[:, k] = (self.doc_term_matrix * self._probability[:, :, k]).sum(axis = 1)
		#self._doc_topic = normalize(self._doc_topic, axis = 1, norm = 'l1')
		denominator = np.sum(self._doc_topic, axis = 1)
		temp = np.zeros(self._numDoc)
		temp[denominator != 0] = np.ones(sum(denominator!=0))/denominator[denominator!=0]
		temp[denominator == 0] = np.ones(sum(denominator == 0)) * 1.0/self.number_of_topic
		self._doc_topic[denominator == 0, :] = np.ones((sum(denominator == 0),self.number_of_topic))
		self._doc_topic = np.dot(np.diag(temp), self._doc_topic)
		
		if self.network:
			# old_loglikelihood = self._old
			new_loglikelihood = self._LogLikelihood()
			if new_loglikelihood < old_loglikelihood:
				while True:
					old_doc_topic = self._doc_topic
					self._doc_topic = (1 - self._gamma) * old_doc_topic + self._gamma * self._adj.dot(old_doc_topic)
					new_loglikelihood = self._LogLikelihood()
					if new_loglikelihood > old_loglikelihood:
						break
					else:
						old_loglikelihood = new_loglikelihood
			for i in range(0, 5):
				old_loglikelihood = new_loglikelihood
				old_doc_topic = self._doc_topic
				self._doc_topic = (1 - self._gamma) * old_doc_topic + self._gamma * self._adj.dot(old_doc_topic)
				new_loglikelihood = self._LogLikelihood()
				if new_loglikelihood < old_loglikelihood:
					self._doc_topic = old_doc_topic
					break
				else:
					old_loglikelihood = new_loglikelihood
		
		self._topic_word += 1e-20
		self._doc_topic += 1e-20



	# calculate the log likelihood
	def _LogLikelihood(self):
		# loglikelihood = 0
		# for i in range(0, self._numDoc):
		# 	for j in range(0, self._numWord):
		# 		# tmp = 0
		# 		# for k in range(0, self.number_of_topic):
		# 		try:
		# 			np.log(self._topic_word[:, j])
		# 		except:
		# 			print 'np.log(self._topic_word[i, :])', np.sum(self._topic_word[i, :] == 0), np.sum(self._topic_word == 0), self._topic_word.shape
		# 			raise
		# 		try:
		# 			np.log(self._doc_topic[i, :])
		# 		except:
		# 			print 'np.log(self._doc_topic[i, :])', np.sum(self._doc_topic[i, :] == 0), np.sum(self._doc_topic == 0), self._doc_topic.shape
		# 			raise
		# 		loglikelihood += self.doc_term_matrix[i, j] * np.dot(self._probability[i, j, :], (np.log(self._topic_word[:, j]) + np.log(self._doc_topic[i, :])))
		# 		# loglikelihood += self.doc_term_matrix[i, j] * tmp


		loglikelihood = 0
		for k in range(0, self.number_of_topic):
			loglikelihood += np.sum(self.doc_term_matrix * self._probability[:, :, k] * \
				np.log(np.outer(self._topic_word[k, :], self._doc_topic[:, k]).T))

		if self.network:
			regular = np.trace(np.dot((self._lap.dot(self._doc_topic)).T, self._doc_topic))
			loglikelihood = (1 - self._lambda) * loglikelihood - self._lambda / 2 * regular
		print loglikelihood
		return loglikelihood

	def RunPLSA(self):
		self._preprocessing()
		if self._save == None:
			print 'random initialize three matrix'
			self._doc_topic = np.random.rand(self._numDoc, self.number_of_topic).astype('f')
			self._topic_word = np.random.rand(self.number_of_topic, self._numWord).astype('f')
			self._probability = np.zeros((self._numDoc, self._numWord, self.number_of_topic), dtype = 'f')

		if self._optimal:
			self._doc_topic = np.zeros((self._numDoc, self.number_of_topic), dtype = 'f')
			for index in range(0, len(self._doc_label)):
				for i in self._doc_label[index]:
					self._doc_topic[index][i] = 1
			self._doc_topic += 1e-10
		
		self._initParameters()

		
		doc_term_matrix = self.doc_term_matrix
		_doc_topic = self._doc_topic
		_word_topic = self._topic_word
		_probability = self._probability

		total_start_time = time.time()
		# until convergence
		for i in range(0, self._maxIteration):
			print 'number of iteration' + '\t' + str(i)
			start_time = time.time()
			# try:
			self._EStep()
			self._MStep()
			self._new = self._LogLikelihood()			
			# except:
			# 	print 'Division by zero! Break the loop'
			# 	break

			doc_term_matrix = self.doc_term_matrix
			_doc_topic = self._doc_topic
			_word_topic = self._topic_word
			_probability = self._probability

			# if(self._old != 1 and abs((self._new - self._old) / self._old) < self._threshold):
			if self._old != 1 and abs(self._new - self._old < 4):
				break
			self._old = self._new

			self._avg_iteration_time[-1] = (self._avg_iteration_time[-1] * i + (time.time() - start_time)) / (i+1)
			print("--- takes %s seconds ---" % (time.time() - start_time))
			print 

		self._avg_iteration_time[-1] = str(self._avg_iteration_time[-1])
		print ("--- %s seconds in total ---" % (time.time() - total_start_time))
		self.doc_term_matrix = doc_term_matrix
		self._doc_topic = _doc_topic
		self._topic_word = _word_topic
		self._probability = _probability

	def print_topic_word_matrix(self, top_n_words):
		# first, sort the words in each topic
		for k in range(0, len(self._topic_word)):
			theta_k = list(self._topic_word[k, :])
			theta_k /= sum(theta_k)
			# make pair (word, index)
			pair = zip(theta_k, range(0, len(theta_k)))
			pair = sorted(pair, key = lambda x: x[0], reverse = True)
			
			print 'topic %d:' % (k)
			for i in range(0, int(top_n_words)):
				print '%20s  \t---\t  %.4f' % (self._CommonWordList[pair[i][1]], pair[i][0])
			print '\n'


	def save_all_data(self, path_to_save):
		with open(path_to_save, 'wb') as outfile:
			pickle.dump(self, outfile)

		# save time complexity information
		with open(path_to_save + '_avg_runtime_in_seconds', 'w') as outfile:
			outfile.write('\t'.join(self._avg_iteration_time))



DEFAULT_DATA_FILE_SUFFIX = "1000"
DEFAULT_RESULT_FILE = "plsa_data_20topics"
DEFAULT_LAMBDA = 0.5
DEFAULT_GAMMA = 0.1
DEFAULT_SYNTHETIC_EDGE_PROB = 0.0

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--data_file_suffix", "-dfs", "data_file_suffix",
	default=DEFAULT_DATA_FILE_SUFFIX,
	help="The suffix of data file")
@click.option("--result_file", "-rf", "result_file",
	default=DEFAULT_RESULT_FILE,
	help="The path of result file")
@click.option("--lambda_par", "-l", "lambda_par",
	default=DEFAULT_LAMBDA,
	help="lambda parameter")
@click.option("--gamma_par", "-g", "gamma_par",
	default=DEFAULT_GAMMA,
	help="gamma parameter")

@click.option("--synthetic_edge_prob", "-sep", "synthetic_edge_prob",
	default=DEFAULT_SYNTHETIC_EDGE_PROB,
	help="synthetic edge probability")

def main(data_file_suffix = DEFAULT_DATA_FILE_SUFFIX, result_file = DEFAULT_RESULT_FILE, lambda_par = DEFAULT_LAMBDA, \
			gamma_par = DEFAULT_GAMMA, synthetic_edge_prob = DEFAULT_SYNTHETIC_EDGE_PROB):
	# np.seterr(all = 'raise')
	np.seterr(divide = 'warn', over = 'warn', under = 'warn',  invalid = 'raise')
	np.random.seed(0)
	doc_path = 'titlesUnderCS_%s.txt' % (data_file_suffix)
	stop_word_path = 'stopwords.txt'
	path_to_adj = 'adjacentMatrixUnderCS_%s' % (data_file_suffix)
	path_to_idname = 'filtered_10_fields.txt' 
	path_to_paperid = 'PaperToKeywords_%s.txt' % (data_file_suffix)

	# Set "network = False" to get a good initialization from PLSA
	plsa = PLSA(doc_path, stop_word_path, path_to_adj, path_to_idname, path_to_paperid, 
					network = True, lambda_par = lambda_par, gamma_par = gamma_par, 
					synthetic_edge_prob = synthetic_edge_prob, optimal = False)
	plsa.RunPLSA()

	# # Run NetPLSA
	# plsa.network = True
	# plsa._old = 1
	# plsa._new = 1
	# plsa._save = not plsa._save
	# plsa.RunPLSA()

	# Print result
	plsa.print_topic_word_matrix(20)
	path_to_save = result_file
	plsa.save_all_data(str(path_to_save))
	

if __name__ == "__main__":
	main()

