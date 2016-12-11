import numpy as np
from sklearn import svm
from collections import defaultdict, Counter
import pickle
import statistics
from netplsa_with_plsa import PLSA
from random import shuffle, sample, randint
from sklearn.model_selection import KFold
from sklearn import linear_model
import click
from sklearn.metrics.cluster import normalized_mutual_info_score
import networkclustering
import re
from scipy.sparse import csr_matrix
import os
from overlapNMI import overlapNMI

np.random.seed(0)

# classification evaluation
def classification(doc_topic, label_list, label_category_list, percentage, accuracy_measure = 'both'):
	label_list = np.array(label_list)
	classifier_list = list()
	doc_index = np.arange(len(label_list))
	kf = KFold(n_splits = 5, shuffle = True)
	train_loose_accuracy = []
	test_loose_accuracy = []
	pred_label_list = [0] * len(label_list)
	for train, test in kf.split(doc_index):
		train_true_label = label_list[train]
		test_true_label = label_list[test]

		# doc -> list of label
		train_label = defaultdict(set)
		test_label = defaultdict(set)
		# label list is doc -> set of label
		label_temp = list()
		for key, val in label_category_list.items():
			label_temp.append(val)

		for i in label_temp:
			current_label = i
			train_label_list = list()

			for index in train:
				if current_label in label_list[index]:
					train_label_list.append(1)
				else:
					train_label_list.append(0)


			X_train = doc_topic[train]
			X_test = doc_topic[test]
		
			model = svm.SVC(kernel='rbf', class_weight = 'balanced')

			model.fit(X_train, train_label_list)
			predict_train = model.predict(X_train)

			for index in range(0, len(predict_train)):
				if predict_train[index] == 1:
					train_label[index].add(current_label)

			predict = model.predict(X_test)
		
			for index in range(0, len(predict)):
				if predict[index] == 1:
					test_label[index].add(current_label)
				else:
					test_label[index]

		if accuracy_measure == 'tight':
			count = 0
			for i in range(0, len(test_label)):
				pred_label_list[test[i]] = test_label[i]
				if test_label[i] == test_true_label[i]:
					count += 1
			print 'test tight accuracy: ' + str(float(count) / len(test_label))
			test_accuracy = float(count) / len(test_label)


			count = 0
			for i in range(0, len(train_label_list)):
				if train_label[i] == train_true_label[i]:
					count += 1
			print 'train tight accuracy: ' + str(float(count) / len(train_label_list))

			return test_accuracy

		elif accuracy_measure == 'loose':
			count = 0
			for i in range(0, len(test_label)):
				pred_label_list[test[i]] = test_label[i]
				if len(test_label[i].intersection(test_true_label[i])) >= 1:
					count += 1
			print 'test loose accuracy: ' + str(float(count) / len(test_label))
			test_accuracy = float(count) / len(test_label)

			count = 0
			for i in range(0, len(train_label_list)):
				if len(train_label[i].intersection(train_true_label[i])) >= 1:
					count += 1
			print 'train loose accuracy: ' + str(float(count) / len(train_label_list))
			return test_accuracy

		else:
			#count_tight = 0
			count_loose = 0
			# print "test lable length" + str(len(test_label))
			for i in range(0, len(test_label)):
				pred_label_list[test[i]] = test_label[i]
				#if test_label[i] == test_true_label[i]:
				#	count_tight += 1
				if len(test_label[i].intersection(test_true_label[i])) >= 1:
					count_loose += 1
			
			#print 'test tight accuracy: ' + str(float(count_tight) / len(test_label))
			print 'test loose accuracy: ' + str(float(count_loose) / len(test_label))

			#test_tight_accuracy = float(count_tight) / len(test_label)
			test_loose_accuracy = np.append(test_loose_accuracy, float(count_loose) / len(test_label))

			#count_tight = 0
			count_loose = 0
			for i in range(0, len(train_label_list)):
				#if train_label[i] == train_true_label[i]:
				#	count_tight += 1
				if len(train_label[i].intersection(train_true_label[i])) >= 1:
					count_loose += 1
			
			#print 'train tight accuracy: ' + str(float(count_tight) / len(train_label_list))
			print 'train loose accuracy: ' + str(float(count_loose) / len(train_label_list))
			train_loose_accuracy.append(float(count_loose) / len(train_label_list))
	#train_loose_accuracy = np.array(train_loose_accuracy)
	#test_loose_accuracy = np.array(test_loose_accuracy)

	# Calculate 
	# label_true_unique = list()
	# label_pred_unique = list()

	# change pred_label_list to matrix
	pred_label_matrix = np.zeros((len(pred_label_list), len(label_category_list)), dtype = 'f')
	for index in range(len(pred_label_list)):
		if len(pred_label_list[index]):
			for item in pred_label_list[index]:
				pred_label_matrix[index][item] = 1
		else:
			pred_label_matrix[index][randint(0, 9)] = 1


	true_label_matrix = np.zeros((len(label_list), len(label_category_list)), dtype = 'f')
	for index in range(0, len(label_list)):
		for i in label_list[index]:
			true_label_matrix[index][i] = 1

	save = (pred_label_matrix, true_label_matrix)
	with open('saved_matrix.pkl', 'wb') as OUTFILE:
		pickle.dump(save, OUTFILE)



	# for i in range(len(label_list)):
	# 	if len(label_list[i]) > 1:
	# 		label_true_unique.append(sample(label_list[i], 1)[0])
	# 	else:
	# 		label_true_unique.append(list(label_list[i])[0])
	# 	if len(pred_label_list[i]) > 1:
	# 		label_pred_unique.append(sample(pred_label_list[i], 1)[0])
	# 	elif len(pred_label_list[i]) == 0:
	# 		label_pred_unique.append(randint(0, 9))
	# 	else:
	# 		# print pred_label_list[i]
	# 		label_pred_unique.append(list(pred_label_list[i])[0])

	# NMI = normalized_mutual_info_score(label_true_unique, label_pred_unique)
	NMI = overlapNMI(true_label_matrix, pred_label_matrix)
	print 'Mean train accuracy: ' + str(statistics.mean(train_loose_accuracy))
	print 'Mean test accuracy: ' + str(statistics.mean(test_loose_accuracy))
	return (statistics.mean(train_loose_accuracy), statistics.mean(test_loose_accuracy), NMI)





# word intrusion evaluation
def preprocessing(doc_path, stop_word_path, lemmatize = True, Stem = False):
	StopWords = set()
	with open(stop_word_path, 'r') as INFILE:
		for line in INFILE.readlines():
			StopWords.add(line.strip())
	row = list()
	col = list()
	data = list()
	wordlist = list()
	
	with open(doc_path, 'r') as INFILE:
		count = 0
		for line in INFILE.readlines():
			temp = Counter(re.split(' ', line.strip()))
			for key, val in temp.items():
				if key in StopWords:
					continue

				if lemmatize:
					try:
						word = str(self._wordnet_lemmatizer.lemmatize(str(self._wordnet_lemmatizer.lemmatize(key)), pos = 'v'))
					except:
						word = key
				elif Stem:
					try:
						word = str(self._lancaster_stemmer.stem(key))
					except:
						word = key
				else:
					word = key
				
				if word in StopWords:
					continue
				
				elif len(word) < 3:
					continue
				
				else:
					if word not in wordlist:
						wordlist.append(word)

				row.append(count)
				'''
				TODO: this part is a little bit slow, which is need to improve the next line
				'''
				col.append(wordlist.index(word))
				data.append(1)
			count += 1
	
	doc_term_matrix = csr_matrix((data, (row, col)), shape = (count, len(wordlist)))
	'''
	TODO: return value is a density matrix, we would like to change it the sparse matrix
	'''
	# saved = (doc_term_matrix.toarray(), count, wordlist)
	saved = (doc_term_matrix, count, wordlist)

	with open('preprocessing_data.pkl', 'wb') as INFILE:
		pickle.dump(saved, INFILE)

	print doc_term_matrix.shape[1]
	return (doc_term_matrix, count, wordlist)


def WordIntrusion(doc_term_matrix, number_of_word, word_topic, \
					number_of_topic = 10, num_topwords = 10, num_instances_per_topic = 50):
	# binary_doc_term = doc_term_matrix.toarray()
	binary_doc_term = doc_term_matrix
	print 'finish dense matrix'
	# binary_doc_term[binary_doc_term > 0] = 1
	# bigram_freq = np.dot(binary_doc_term.T, binary_doc_term)
	bigram_freq = (doc_term_matrix.transpose()).dot(doc_term_matrix)
	bigram_freq = bigram_freq.toarray()
	
	print 'finsh bigram freq matrix'

	unigram_prob = (1.0 * np.diag(bigram_freq)) / np.trace(bigram_freq)
	bigram_prob = (1.0 * bigram_freq) / (np.sum(bigram_freq) - np.trace(bigram_freq)) + 1e-10
	PMI_denominator = np.outer(unigram_prob, unigram_prob)
	PMI_mat = np.log(bigram_prob) - np.log(PMI_denominator)
	PMI = PMI_mat.sum(axis = 0)
	CP1 = (PMI_mat / unigram_prob).sum(axis = 0)
	CP2 = ((1.0 / unigram_prob).T * PMI_mat).sum(axis = 0)
	# change 3 column vector to a n * 3 matrix
	word_intrusion_features = np.column_stack((PMI, CP1, CP2))
	
	ind_words = np.ndarray((number_of_topic, len(word_topic)))
	ind_topwords = np.ndarray((number_of_topic, num_topwords))
	num_instances = num_instances_per_topic * number_of_topic
   
	train_accuracy = np.zeros(number_of_topic)
	test_accuracy = np.zeros(number_of_topic)
	
	for k in range(number_of_topic):
		# get the indices that will sort the array, from index of the smallest number to that of the largest
		ind_words[k, :] = np.argsort(word_topic[:, k])
		# print 'index sort in K topic'
		# print ind_words[k, :]
		# # get the indices of the top 10 words in each topic
		# print 'top 10 index'
		ind_topwords[k, :] = ind_words[k, :][::-1][:num_topwords]  
		
	for k in range(number_of_topic):
		#randomly sample 5 words from the top 10 words in each topic
		instances = np.ndarray((num_instances_per_topic, 18))
		responses = np.ndarray((num_instances_per_topic, 6))
		
		other_topwords_ind = np.unique(np.delete(ind_topwords, k, axis = 0).ravel())
		last_50_ind = ind_words[k, :int(np.floor(0.5 * number_of_word))]
		intersection = np.intersect1d(other_topwords_ind, last_50_ind)
		
		# print intersection

		if intersection.size > num_instances_per_topic:
			for i in range(num_instances_per_topic):
				topwords_ind = np.random.choice(ind_topwords[k, :], int(0.5 * num_topwords))
				indices_of_instances = np.append(topwords_ind, intersection[i]).astype(int) # len=6
				instances[i, :] = word_intrusion_features[indices_of_instances, :].ravel()
				responses[i, :] = word_topic[indices_of_instances, k]
		predict_train = np.ndarray((6, (num_instances_per_topic / 2)))
		predict_test = np.ndarray((6, (num_instances_per_topic / 2)))
		
		for i in range(6):
			new_response = responses[:, i] # size 10*1
			dataframe = np.column_stack((instances, new_response))
			np.random.shuffle(dataframe)
			train_set = dataframe[0 : (num_instances_per_topic / 2), :]
			test_set = dataframe[(num_instances_per_topic / 2):, :]
					
			regr = linear_model.LinearRegression()
			regr.fit(train_set[:, :-1], train_set[:, -1])
			predict_train[i, :] = regr.predict(train_set[:, :-1])
			predict_test[i, :] = regr.predict(test_set[:, :-1])
			
		# print 'predict train'
		# print predict_train
		# print 'predict test'
		# print predict_test

		detected_intruder_ind_train = np.zeros((num_instances_per_topic / 2))
		detected_intruder_ind_test = np.zeros((num_instances_per_topic / 2))

		count_train = 0
		count_test = 0

		for j in range((num_instances_per_topic / 2)):
			detected_intruder_ind_train[j] = predict_train[:, j].argsort()[0]
			detected_intruder_ind_test[j] = predict_test[:, j].argsort()[0]
			
			# print 'intrusion train'
			# print detected_intruder_ind_train
			# print 'intrusion test'
			# print detected_intruder_ind_test

			if int(detected_intruder_ind_train[j]) == 5:
				count_train += 1
			if int(detected_intruder_ind_test[j]) == 5:
				count_test += 1

		# print 'train'
		# print detected_intruder_ind_train
		# print 'test'
		# print detected_intruder_ind_test
		# print '\n'

		train_accuracy[k] = float(count_train) / (num_instances_per_topic * 0.5)
		test_accuracy[k] = float(count_test) / (num_instances_per_topic * 0.5)

	print 'Word Intrusion Detection Train Accuracy for Each Topic'
	print train_accuracy
	print 'Word Intrusion Detection Test Accuracy for Each Topic'
	print test_accuracy


DEFAULT_SOURCE_FILE = "plsa_data"
DEFAULT_RESULT_FILE = "plsa_data_evaluation"

# CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
# @click.command(context_settings=CONTEXT_SETTINGS)
# @click.option("--source_file", "-sf", "source_file",
# 	default=DEFAULT_SOURCE_FILE,
# 	help="The path of source file")
# @click.option("--result_file", "-rf", "result_file",
# 	default=DEFAULT_RESULT_FILE,
# 	help="The path of result file")

def evaluation_classification(source_file = DEFAULT_SOURCE_FILE, result_file = DEFAULT_RESULT_FILE):
	data_file = str(source_file)
	with open(data_file, 'r') as INFILE:
		data = pickle.load(INFILE)
	# data_percentage = [0.8]
	# for i in data_percentage:
	# i is the data_percentage
	data_percentage = 0.8
	print 'data_percentage: ' + str(data_percentage)
	with open(str(result_file), 'w') as res_file:
		res_file.write('train_loose_accuracy\ttest_loose_accuracy\n')
		train_loose_accuracy, test_loose_accuracy, NMI = classification(data.doc_term_matrix, \
					data._doc_label, data._label_category, data_percentage)
		res_file.write('%f\t%f' % (train_loose_accuracy, test_loose_accuracy))
	print 'The NMI between topic model and true label is\t' + str(NMI)
	return NMI

def evaluation_word_intrusion(doc_path = 'titlesUnderCS_10000.txt', stop_word_path = 'stopwords.txt', \
								lemmatize = True, Stem = False, source_file = DEFAULT_SOURCE_FILE):
	if not os.path.exists('preprocessing_data_20topic.pkl'):
		doc_term_matrix, number_of_doc, wordlist = preprocessing(doc_path, stop_word_path, \
				lemmatize = lemmatize, Stem = Stem)
	else:
		print 'load data'
		with open('preprocessing_data_20topic.pkl', 'rb') as INFILE:
			temp = pickle.load(INFILE)
			doc_term_matrix = temp[0]
			number_of_topic = temp[1]
			wordlist = temp[2]

	data_file = str(source_file)
	with open(data_file, 'r') as INFILE:
		data = pickle.load(INFILE)
	word_topic = data._topic_word.T

	accuracy = WordIntrusion(doc_term_matrix, len(wordlist), word_topic)

def evaluation_clustering(source_file = DEFAULT_SOURCE_FILE, input_file = 'adjacentMatrixUnderCS_1000'):
	data_file = str(source_file)
	with open(data_file, 'r') as INFILE:
		data = pickle.load(INFILE)

	NMI_network = networkclustering.networkClustering(input_file, data)

	return NMI_network


if __name__ == "__main__":
	# NMI_topic_network = evaluation_classification()
	# NMI_network = evaluation_clustering()
	evaluation_word_intrusion()


