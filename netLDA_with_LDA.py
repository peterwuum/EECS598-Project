import time
from collections import Counter, defaultdict
import re
import numpy as np
from scipy.sparse.csgraph import laplacian
from sklearn.preprocessing import normalize
from scipy.special import psi, gammaln, polygamma

import csv
import pickle

class LDA(object):
	def __init__(self, doc_path, stop_word_path, path_to_adj, path_to_idname, path_to_paperid,
					number_of_topic = 10, max_document, batch_size, 
					tau, kappa, alpha = 0.05, eta = 0.05, threshold = 1e-3, maxIteration = 1000, 
					online = True, auto_set_hyperparameter = False, update_alpha = False, update_eta = False,
					network = False, lemmatize = True, stemmer = False):
		
		'''
		notation:
		theta ~ Dirichlet(alpha)
		beta ~ Dirichlet(eta)
		gamma: document topic matrix (document (batch size), number of topic)
		lambda: topic word matrix (number of topics, number of word in commonwordlist)
		'''

		if online and (update_alpha or update_eta):
			raise SystemExit('Online Latent Dirichlet Allocation could not update alpha or eta')


		self._doc_path = doc_path
		self._stopword = set()
		with open(stop_word_path, 'r') as INFILE:
			for line in INFILE.readlines():
				self._stopword.add(line.strip())

		with open(path_to_adj, 'r') as INFILE:
			self._adj = pickle.load(INFILE)
		
		self._lap = laplacian(self._adj)
		self._adj = normalize(self._adj, norm='l1', axis=1)

		self._docname_to_id = dict()
		name2id = 0
		with open(path_to_idname, 'r') as INFILE:
			for line in INFILE.readlines():
				self._docname_to_id[re.split('\t', line)[0]] = name2id
				name2id += 1

		# doc_label is a list of set of documents label ids
		self._doc_label = list()
		with open(path_to_paperid, 'r') as INFILE:
			for line in INFILE.readlines():
				ids = re.split('\t', line.strip())
				temp = set()
				for item in ids:
					temp.add(item)
				self._doc_label.append(temp)

		self.number_of_topic = number_of_topic
		# maximum document to be seen
		self._document = max_document
		self._batch = batch_size
		self._maxIteration = maxIteration
		self._threshold = threshold
		self._online = online

		self._CommonWordList = list()

		# parameter for online LDA
		self._tau = tau + 1
		self._kappa = kappa
		self._t = 0

		if auto_set_hyperparameter:
			alpha = float(1) / self._topic
			eta = float(1) / self._topic

		self._alpha = np.asarray([alpha for i in range(0, self._topic)])
		# update alpha parameter
		self._update_alpha = update_alpha	
		self._eta = (np.asarray([eta for i in range(0, self._topic)])).reshape((self._topic, 1))
		# update eta parameter
		self._update_eta = update_eta

		# parameters for LDA init
		self.doc_term_matrix = 0
		self._doc_topic = 0
		self._topic_word = 0
		self._numDoc = 0
		self._numWord = 0
		self._logbeta = 0
		self._beta = 0

		self._wordnet_lemmatizer = False
		if lemmatize:
			self._wordnet_lemmatizer = WordNetLemmatizer()
		
		self._lancaster_stemmer = False
		if stemmer:
			self._lancaster_stemmer = LancasterStemmer()

		self._network = network

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
			if word_entropy[i] < -1 and len(_CommonWordListTmp) < self._numSelectedWord:
				_CommonWordListTmp.append(self._CommonWordList[i])
				_indexTmp.append(i)

		self._CommonWordList = _CommonWordListTmp
		_CommonWordListSet = set(self._CommonWordList)


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

		print 'Built processed adjacent matrix with size (%d, %d)' % self.doc_term_matrix.shape

		# transformer = TfidfTransformer(smooth_idf = False)
		# self.doc_term_matrix = transformer.fit_transform(self.doc_term_matrix).toarray()
		# print self.doc_term_matrix

	def _initParameters(self):
		if not self._online:
			self._batch = self._numDoc
		# initialize the lambda matrix (topic, word)
		self._topic_word = np.random.gamma(100.0, 1.0/100.0, (self._topic, self._numWord)) 
		self._logbeta = self._DirichletExpectation(self._topic_word)
		self._beta = np.exp(self._logbeta)
		# initialize the gamma matrix (document, topic)
		self._doc_topic = np.zeros((self._numDoc, self._topic))

	# function for calculating the Dirichlet Expectation
	def _DirichletExpectation(self, alpha):
		if (len(alpha.shape) == 1):
			# without change the size
			return psi(alpha) - psi(np.sum(alpha))
		# change to the same size
		return psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]

	# two functions for get and copy the data from and to the original one
	def _fetch_matrix(self, start, end):
		result = list()
		if (start < end):
			result = self.doc_term_matrix[start : end]
		else:
			result = self.doc_term_matrix[start : self._numDoc]

			for i in range(0, end):
				result.append(self.doc_term_matrix[i])
			result = np.asarray(result)

		return result

	# the original matrix: one of self matrix
	def _save_matrix(self, saving_matrix, start, end):
		if (start < end):
			self._doc_topic[start : end] = saving_matrix
		else:
			self._doc_topic[start : self._numDoc] = saving_matrix[0 : (self._numDoc - start)]
			self._doc_topic[0 : end] = saving_matrix[(self._numDoc - start) : len(saving_matrix)]
	
	def _hyperparameter_estimation(self, prior, N, functionhat, rho):
		temp_prior = np.copy(prior)
		gradient_function = N * (psi(np.sum(prior)) - psi(prior) + functionhat)

		a = N * polygamma(1, np.sum(prior))
		b = -N * polygamma(1, prior)
		c = np.sum(gradient_function / b) / (1 / a + np.sum(1 / b))
		temp_prior = -(gradient_function - c) / b

		# calculate the prior
		if all(prior + rho * temp_prior) > 0:
			prior += rho * temp_prior

		return prior

	def _alpha_update(self, gamma_matrix):
		
		N = float(len(gamma_matrix))
		functionhat = sum(self._DirichletExpectation(it) for it in gamma_matrix) / N
		self._alpha = self._hyperparameter_estimation(self._alpha, N, functionhat, self._rhot)
		print 'self._alpha = ' + str(self._alpha)
		return self._alpha

	def _eta_update(self, lambda_matrix):
		# need to make sure self.eta should be a column vector self._eta shape is 1, N
		
		N = float(lambda_matrix.shape[1])
		functionhat = sum(self._DirichletExpectation(it) for it in lambda_matrix.transpose()) / N
		# reshape the eta to (number of topic, 1) row vector
		functionhat = functionhat.reshape((self._topic, 1))
		self._eta = self._hyperparameter_estimation(self._eta, N, functionhat, self._rhot)
		print 'self._eta = ' + str(self._eta)
		return self._eta
	



	# TODO: need to change the format of EM algorithm
	# _EStep _Mstep _LogLikelihood

	
	# get the document - topic distribution matrix without updating the hyperparameter and topic - word matrix
	def Inference(self, wordindex, wordcount):
		# temp gamma is initialized in the E_step function
		corpus_size = len(wordcount)
		gamma = np.random.gamma(100.0, 1.0 / 100.0, (corpus_size, self._topic))
		logtheta = self._DirichletExpectation(gamma)
		theta = np.exp(logtheta)

		sufficient_stats = np.zeros(self._topic_word.shape)

		# get gamma (document - topic matrix) for each document
		diff = 0
		for d in range(0, corpus_size):
			index = wordindex[d]
			count = wordcount[d]
			gamma_d = gamma[d, :]
			logtheta_d = logtheta[d, :]
			theta_d = theta[d, :]
			beta_d = self._beta[:, index]
			phi = np.dot(theta_d, beta_d) + 1e-100
			# update gamma and phi until convergence
			for i in range(0, self._inference_iteration):
				old_gamma = gamma_d
				gamma_d = self._alpha + theta_d * np.dot(count / phi, beta_d.T)
				logtheta_d = self._DirichletExpectation(gamma_d)
				theta_d = np.exp(logtheta_d)
				phi = np.dot(theta_d, beta_d) + 1e-100
				# If gamma hasn't changed much, we're done.
				diff = np.mean(abs(gamma_d - old_gamma))
				if (diff < self._threshold):
					break
				# update gamma
			gamma[d, :] = gamma_d
			sufficient_stats[:, index] += np.outer(theta_d.T, count / phi)
		
		sufficient_stats *= self._beta
		return gamma, sufficient_stats

	# do e step which will be used in m step function
	def _E_Step(self, wordindex, wordcount):
		gamma, sufficient_stats = self.Inference(wordindex, wordcount)
		return gamma, sufficient_stats

	def _perplexity(self, wordindex, wordcount, gamma, total_document = None):
		corpus_word = sum(count for document in wordcount for count in document)
		if total_document == None:
			sample_rate = self._document * 1.0 / len(wordcount)
		else:
			sample_rate = total_document * 1.0 / len(wordcount)

		# note: change perword_bound to perplexity
		perplexity = self._approximate(wordindex, wordcount, gamma, subsampling = sample_rate / (sample_rate * corpus_word))
		# perplexity = np.exp2(-perword_bound)
		return perplexity


	def _approximate(self, wordindex, wordcount, gamma, subsampling = 1.0):
		loss = 0
		logtheta = self._DirichletExpectation(gamma)
		theta = np.exp(logtheta)

		# according to the equation 4
		for d in range(0, self._batch):
			gamma_d = gamma[d, :]
			# the word index for document d
			index = wordindex[d]
			count = np.array(wordcount[d])
			phi = np.zeros(len(index))
			# iterate the word in that document
			for i in range(0, len(index)):
				# phi[i] = np.log(np.sum(np.exp(logtheta[d, :] + self._logbeta[:, index[i]]))) 
				phi[i] = np.log(sum(np.exp(logtheta[d, :] + self._logbeta[:, index[i]] - max(logtheta[d, :] + self._logbeta[:, index[i]])))) + max(logtheta[d, :] + self._logbeta[:, index[i]])
			loss += np.sum(count * phi)
			loss += np.sum((self._alpha - gamma_d) * logtheta[d, :])
			loss += np.sum(gammaln(gamma_d) - gammaln(self._alpha))
			loss += gammaln(np.sum(self._alpha)) - gammaln(np.sum(gamma_d))

		# tradition version of update loss function
		# loss += np.sum((self._alpha - gamma) * logtheta)
		# loss += np.sum(gammaln(gamma) - gammaln(self._alpha))
		# loss += sum(gammaln(self._alpha * self._topic) - gammaln(np.sum(gamma, 1)))
		
		loss = loss * subsampling
		loss += np.sum((self._eta - self._topic_word) * self._logbeta)
		loss += np.sum(gammaln(self._topic_word) - gammaln(self._eta))
		# self._eta is a scalar
		if np.ndim(self._eta) == 0:
			loss += np.sum(gammaln(self._eta * self._numWord) - gammaln(np.sum(self._topic_word, 1)))
		else:
			loss += np.sum(gammaln(np.sum(self._eta, 1)) - gammaln(np.sum(self._topic_word, 1)))
		# loss += np.sum(gammaln(np.sum(self._eta, 1)) - gammaln(np.sum(self._topic_word, 1)))
		return loss




	def _M_Step(self, wordindex, wordcount):
		self._rhot = pow(self._tau + self._t, -self._kappa)
		# update lamda.
		gamma, sufficient_stats = self._E_Step(wordindex, wordcount)
		# update alpha based on the new gamma (just based on partial data not the whole dataset)
		if self._update_alpha:
			self._alpha = self._alpha_update(gamma)
		# Estimate held-out likelihood for current values of lambda.(perplexity)
		# loss = self._Loss_Function(wordindex, wordcount, gamma)
		perplexity = self._perplexity(wordindex, wordcount, gamma, total_document = None)
		# Update lambda based on documents.
		self._topic_word = (1 - self._rhot) * self._topic_word + self._rhot * (self._eta + self._document * sufficient_stats / self._batch)
		self._logbeta = self._DirichletExpectation(self._topic_word)
		self._beta = np.exp(self._logbeta)
		self._t += 1
		
		# update eta based on the new lambda
		if self._update_eta:
			self._eta = self._eta_update(self._topic_word)

		return gamma, perplexity

	def run_topic_model(self):
		self._preprocessing()
		self._initParameters()
		count = 0
		perplexity_store = 0
		if self._online:
			while count + self._batch <= self._document:
				temp_corpus_matrix = self._fetch_matrix(count % self._numDoc, (count + self._batch) % self._numDoc)
				wordindex = list()
				wordcount = list()
				for doc_word_list in temp_corpus_matrix:
					temp_index = list()
					temp_count = list()

					for i, val in enumerate(doc_word_list): 
						if (val != 0):
							temp_index.append(i)
							temp_count.append(val)
					wordindex.append(temp_index)
					wordcount.append(temp_count)
				gamma, perplexity = self._M_Step(wordindex, wordcount)
				self._save_matrix(gamma, count % self._numDoc, (count + self._batch) % self._numDoc)

				count += self._batch
				if count < self._numDoc:
					perplexity_store = perplexity
				else:
					if abs((perplexity - perplexity_store) / perplexity_store) < self._threshold:
						break
					else:
						perplexity_store = perplexity

		else:
			wordindex = list()
			wordcount = list()
			for doc_word_list in self.doc_term_matrix:
				temp_index = list()
				temp_count = list()
				for i, val in enumerate(doc_word_list): 
					if (val != 0):
						temp_index.append(i)
						temp_count.append(val)
				wordindex.append(temp_index)
				wordcount.append(temp_count)
			while count + self._batch <= self._document:
				gamma, perplexity = self._M_Step(wordindex, wordcount)
				self._doc_topic = gamma

				count += self._batch
				if count < self._numDoc:
					perplexity_store = perplexity
				else:
					if abs((perplexity - perplexity_store) / perplexity_store) < self._threshold:
						break
					else:
						perplexity_store = perplexity


	def print_topic_word_matrix(self, top_n_words):
		if top_n_words > len(self._CommonWordList):
			raise SystemExit('Please input n less than' + ' ' + str(len(self._CommonWordList)))
		# first, sort the words in each topic
		for k in range(0, len(self._topic_word)):
			lambda_k = list(self._topic_word[k, :])
			lambda_k /= sum(lambda_k)
			# make pair (word, index)
			pair = zip(lambda_k, range(0, len(lambda_k)))
			pair = sorted(pair, key = lambda x: x[0], reverse = True)
			
			print 'topic %d:' % (k)
			for i in range(0, int(top_n_words)):
				print '%20s  \t---\t  %.4f' % (self._CommonWordList[pair[i][1]], pair[i][0])
			print '\n'

	def print_doc_topic_matrix(self, min_probability = 0.01):
		min_probability = 1.0 / self._topic
		for k in range(0, len(self._doc_topic)):
			gamma_k = list(self._doc_topic[k, :])
			gamma_k /= sum(gamma_k)
			pair = zip(gamma_k, range(0, len(gamma_k)))
			pair = sorted(pair, key = lambda x: x[0], reverse = True)

			print 'document' + ' ' + str(k) + ' ' + '=',
			for i in range(0, self._topic):
				if (pair[i][0] >= min_probability):
					if (i != self._topic - 1):
						print str(pair[i][0]) + '-' + 'topic' + str(pair[i][1]) + ' + ',
					else:
						print str(pair[i][0]) + '-' + 'topic' + str(pair[i][1]),
			print '\n'


	def get_topic_word_matrix(self):
		return self._topic_word

	def get_doc_topic_matrix(self):
		return self._doc_topic

	def get_word_list(self):
		return self._CommonWordList
	
	def save_as_csv(self, path_to_save, header, output):
		with open(path_to_save, 'w') as outfile:
			writer = csv.DictWriter(outfile, header)
			writer.writeheader()
			for row in output_rows:
				writer.writerow(row)

	def save_topic_word_matrix(self, path_to_save):
		output_rows = []
		row = []
		header = ['\s']
		header.append(word for word in self._CommonWordList)
		# Matrix to list of list
		for i in range(0, len(self._topic_word)):
			row.append('topic_' + str(i))
			row.append(val for val in list(self._topic_word[i, :]))
			output_rows.append(row)

		# save as csv
		self.save_as_csv(path_to_save, header, output_rows)
		
		
	def save_doc_topic_matrix(self, path_to_save):
		output_rows = []
		row = []
		header = ['\s']
		header.append('topic_' + str(item) for itme in range(0, self._topic))
		# Matrix to list of list
		for i in range(0, len(self._doc_topic)):
			row.append('document_' + str(i))
			row.append(val for val in list(self._doc_topic[i, :]))
			output_rows.append(row)

		# save as csv
		self.save_as_csv(path_to_save, header, output_rows)

	def save_all_data(self, path_to_save):
		with open(path_to_save, 'w') as outfile:
			pickle.dump(self, outfile)

	def feature_for_new_documents(self, corpus_matrix, update = False):
		number_of_document = len(corpus_matrix)
		wordindex = list()
		wordcount = list()
		for doc_word_list in corpus_matrix:
			temp_index = list()
			temp_count = list()
			for i, val in enumerate(doc_word_list): 
				if (val != 0):
					temp_index.append(i)
					temp_count.append(val)
			wordindex.append(temp_index)
			wordcount.append(temp_count)

		if not update:
			# just return the topic feature of the new document
			return self.Inference(wordindex, wordcount)[0]
		else:
			# need to update the topic word matrix
			return self._M_Step(wordindex, wordcount)[0]

			