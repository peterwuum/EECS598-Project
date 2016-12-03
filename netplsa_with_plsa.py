from collections import Counter, defaultdict
import re
import numpy as np
from scipy.sparse.csgraph import laplacian
from sklearn.preprocessing import normalize

from scipy.sparse import csr_matrix
import pickle
from sklearn.feature_extraction.text import TfidfTransformer

# TODO: change the EM to distributed version

class PLSA(object):
	def __init__(self, doc_path, stop_word_path, path_to_adj, path_to_idname, path_to_paperid, number_of_topic = 10, maxIteration = 30, 
			threshold = 0.02, network = False, lambda_par = 0.5, gamma_par = 0.1):
		self._doc_path = doc_path
		self._stopword = set()
		with open(stop_word_path, 'r') as INFILE:
			for line in INFILE.readlines():
				self._stopword.add(line.strip())
		
		self.number_of_topic = number_of_topic
		self._maxIteration = maxIteration
		self._threshold = threshold
		self._CommonWordList = list()

		self.doc_term_matrix = 0
		self._doc_topic = 0
		self._word_topic = 0
		self._probability = 0
		self._numDoc = 0
		self._numWord = 0

		with open(path_to_adj, 'r') as INFILE:
			self._adj = pickle.load(INFILE)
		
		self._lap = laplacian(self._adj)
		self._adj = normalize(self._adj, norm='l1', axis=1)
		self._network = network
		self._lambda = lambda_par
		self._gamma = gamma_par

		self._old = 1
		self._new = 1

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


	def _preprocessing(self):
		list_of_doc_word_count = list()
		# document frequency
		word_doc_list = defaultdict(int)
		
		with open (self._doc_path, 'r') as INFILE:
			for line in INFILE.readlines():
				temp = Counter(re.split(' ', line.strip()))
				for key, val in temp.items():
					if key in self._stopword:
						temp.pop(key)
						continue
					elif len(key) < 3:
						temp.pop(key)
						continue
					else:
						word_doc_list[key] += 1
						if key not in self._CommonWordList:
							self._CommonWordList.append(key)
							# print key
				list_of_doc_word_count.append(temp)

		self._numDoc = len(list_of_doc_word_count)
		print 'document'
		print self._numDoc
		
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

		self._numWord = len(self._CommonWordList)
		print 'word'
		print self._numWord
		
		self.doc_term_matrix = np.zeros(shape = (self._numDoc, self._numWord))
		print 'finish build matrix'
		
		for i in range(0, len(list_of_doc_word_count)):
			for key, val in list_of_doc_word_count[i].items():
				if key in self._CommonWordList:
					self.doc_term_matrix[i][self._CommonWordList.index(key)] = val
		
		# transformer = TfidfTransformer(smooth_idf = False)
		# self.doc_term_matrix = transformer.fit_transform(self.doc_term_matrix).toarray()
		print self.doc_term_matrix

	def _initParameters(self):
		normalization = np.sum(self._doc_topic, axis = 1)
		for i in range(0, len(normalization)):
			self._doc_topic[i] /= normalization[i]

		normalization = np.sum(self._word_topic, axis = 1)
		for i in range(0, len(normalization)):
			self._word_topic[i] /= normalization[i]

	
	def _EStep(self):
		for i in range(0, self._numDoc):
			for j in range(0, self._numWord):
				denominator = 0;
				for k in range(0, self.number_of_topic):
					self._probability[i, j, k] = self._word_topic[k, j] * self._doc_topic[i, k];
					denominator += self._probability[i, j, k];
				if denominator == 0:
					for k in range(0, self.number_of_topic):
						self._probability[i, j, k] = 0;
				else:
					for k in range(0, self.number_of_topic):
						self._probability[i, j, k] /= denominator;

	def _MStep(self):
		old_loglikelihood = self._LogLikelihood()
		# update topic-word matrix
		for k in range(0, self.number_of_topic):
			denominator = 0
			for j in range(0, self._numWord):
				self._word_topic[k, j] = 0
				for i in range(0, self._numDoc):
					self._word_topic[k, j] += self.doc_term_matrix[i, j] * self._probability[i, j, k]
			
			denominator = np.sum(self._word_topic[k,:])
			if denominator == 0:
				self._word_topic[k, :] = 1.0 / self._numWord
			else:
				self._word_topic[k, :] /= denominator
		# update document-topic matrix
		for i in range(0, self._numDoc):
			denominator = 0
			for k in range(0, self.number_of_topic):
				self._doc_topic[i, k] = 0
				for j in range(0, self._numWord):
					self._doc_topic[i, k] += self.doc_term_matrix[i, j] * self._probability[i, j, k]
			
			denominator = np.sum(self._doc_topic[i,:])
			if denominator == 0:
				self._doc_topic[i,:] = 1.0 / self.number_of_topic
			else:
				self._doc_topic[i,:] /= denominator
		
		if self._network:
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


	# calculate the log likelihood
	def _LogLikelihood(self):
		loglikelihood = 0
		for i in range(0, self._numDoc):
			for j in range(0, self._numWord):
				tmp = 0
				for k in range(0, self.number_of_topic):
					tmp += self._probability[i, j, k] * (np.log(self._word_topic[k, j]) + np.log(self._doc_topic[i, k]))
				loglikelihood += self.doc_term_matrix[i, j] * tmp

		if self._network:
			regular = np.trace(np.dot((self._lap.dot(self._doc_topic)).T, self._doc_topic))
			loglikelihood = (1 - self._lambda) * loglikelihood - self._lambda / 2 * regular
		print loglikelihood
		return loglikelihood

	def RunPLSA(self):
		self._preprocessing()
		self._doc_topic = np.random.rand(self._numDoc, self.number_of_topic)
		self._word_topic = np.random.rand(self.number_of_topic, self._numWord)
		self._probability = np.zeros((self._numDoc, self._numWord, self.number_of_topic))
		self._initParameters()
		
		# until convergence
		for i in range(0, self._maxIteration):
			print 'number of iteration' + '\t' + str(i)
			self._EStep()
			self._MStep()
			self._new = self._LogLikelihood()			
			if(self._old != 1 and abs((self._new - self._old) / self._old) < self._threshold):
				break
			self._old = self._new

	def print_topic_word_matrix(self, top_n_words):
		# first, sort the words in each topic
		for k in range(0, len(self._word_topic)):
			theta_k = list(self._word_topic[k, :])
			theta_k /= sum(theta_k)
			# make pair (word, index)
			pair = zip(theta_k, range(0, len(theta_k)))
			pair = sorted(pair, key = lambda x: x[0], reverse = True)
			
			print 'topic %d:' % (k)
			for i in range(0, int(top_n_words)):
				print '%20s  \t---\t  %.4f' % (self._CommonWordList[pair[i][1]], pair[i][0])
			print '\n'


	def save_all_data(self, path_to_save):
		with open(path_to_save, 'w') as outfile:
			pickle.dump(self, outfile)

if __name__ == '__main__':
	doc_path = 'titlesUnderCS.txt'
	stop_word_path = 'stopwords.txt'
	path_to_adj = 'adjacentMatrixUnderCS'
	path_to_idname = 'filtered_10_fields.txt' 
	path_to_paperid = 'PaperToKeyWords.txt'
	plsa = PLSA(doc_path, stop_word_path, path_to_adj, path_to_idname, path_to_paperid)
	plsa.RunPLSA()
	plsa.print_topic_word_matrix(20)
	path_to_save = 'plsa_data'
	plsa.save_all_data(path_to_save)
