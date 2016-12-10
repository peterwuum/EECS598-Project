import numpy as np
from sklearn import svm
from collections import defaultdict
import pickle
import statistics
from netplsa_with_plsa import PLSA
from random import shuffle, sample, randint
from sklearn.model_selection import KFold
import click
from sklearn.metrics.cluster import normalized_mutual_info_score
import networkclustering


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
	label_true_unique = list()
	label_pred_unique = list()
	# print pred_label_list
	for i in range(len(label_list)):
		if len(label_list[i]) > 1:
			label_true_unique.append(sample(label_list[i], 1)[0])
		else:
			label_true_unique.append(list(label_list[i])[0])
		if len(pred_label_list[i]) > 1:
			label_pred_unique.append(sample(pred_label_list[i], 1)[0])
		elif len(pred_label_list[i]) == 0:
			label_pred_unique.append(randint(0, 9))
		else:
			# print pred_label_list[i]
			label_pred_unique.append(list(pred_label_list[i])[0])

	NMI = normalized_mutual_info_score(label_true_unique, label_pred_unique)
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
					if word not in wordlist():
						wordlist.append(word)

				row.append(count)
				# this part is a little bit slow, which is need to improve the next line
				col.append(wordlist.index(word))
				data.append(val)
			count += 1
	
	doc_term_matrix = csr_matrix((data, (row, col)), shape = (count, len(wordlist)))
	# return value is a density matrix, we would like to change it the sparse matrix
	return (doc_term_matrix.toarray(), count, wordlist) 

# def WordIntrusion(doc_term_matrix, number_of_topic = 10, number_of_word, word_topic, num_topwords = 10, intruder_ind = 200, num_instances_per_topic = 10):
# 	#unigram_freq = self.doc_term_matrix.sum(axis = 0)
# 	#unigram_prob = unigram_freq / np.sum(self.doc_term_matrix)
# 	binary_doc_term = doc_term_matrix
# 	binary_doc_term[binary_doc_term > 0] = 1
# 	bigram_freq = np.dot(binary_doc_term.T, binary_doc_term)
# 	unigram_prob = np.diag(bigram_freq) / np.trace(bigram_freq)
# 	#np.fill_diagonal(bigram_freq, 0)
# 	bigram_prob = bigram_freq / (np.sum(bigram_freq) - np.trace(bigram_freq))
# 	PMI_denominator = np.outer(unigram_prob, unigram_prob)
# 	PMI_mat = np.log(bigram_prob / PMI_denominator)
# 	PMI = PMI_mat.sum(axis=0)
# 	CP1 = (PMI_mat.dot(1.0 / unigram_prob)).sum(axis=0)
# 	CP2 = ((1.0 / unigram_prob).dot(PMI_mat)).sum(axis=0)
# 	# change 3 column vector to a n * 3 matrix
# 	word_intrusion_features = np.column_stack((PMI, CP1, CP2))
	
# 	ind_words = np.ndarray((number_of_topic, number_of_word))
# 	ind_topwords = np.ndarray((number_of_topic, num_topwords))
# 	num_instances = num_instances_per_topic * number_of_topic
# 	instances = np.ndarray((num_instances, 6))
# 	responses = np.zeros(num_instances)
# 	for k in range(number_of_topic):
# 		# get the indices that will sort the array, from index of the smallest number to that of the largest
# 		ind_words[k, :] = np.argsort(word_topic[:, k])
# 		# get the indices of the top 10 words in each topic
# 		ind_topwords[k, :] = ind_words[k, :][::-1][:num_topwords]
		
# 		#topwords_feature = word_intrusion_features[ind_topwords[k, :], :]
# 		#intruder_feature = word_intrusion_features[intruder_ind, ]
		
		
# 	for k in range(number_of_topic):
# 		#for i in range(num_instances_per_topic):
# 			# randomly sample 5 words from the top 10 words in each topic
		
# 		other_topwords_ind = np.delete(ind_topwords, k, axis = 0).ravel()
# 		last_50_ind = ind_words[k, :np.floor(0.5 * number_of_word)]
# 		intersection = np.interset1d(other_topwords_ind, last_50_ind)
# 		if intersection.size > 10:
# 			for i in range(num_instances_per_topic):
# 				topwords_ind = np.random.choice(ind_topwords[k, :], 0.5 * num_topwords)
# 				indices_of_instances = np.append(topwords_ind, intersection[i])
# 				instances[k * num_instances_per_topic + i, :] = \
# 					word_intrusion_features[indices_of_instances, :]
# 				responses[k * num_instances_per_topic + i] = \
# 					word_topic[indices_of_instances, k]
			
# #            for j in range(number_of_topic):
# #                if j != i:
# #                    # Check if any top word in some other topic j rank in
# #                    # the last 50% in the current topic i. If found, break the loop.
# #                    intersection = np.intersect1d(ind_words[k, :np.floor(0.5*number_of_word)], 
# #                                                  ind_topwords[j,:])
# #                    if intersection.size != 0:
# #                        indices_of_instances = np.append(topwords_ind, intersection[0])
# #                        instances[k*num_instances_per_topic + i, :] = \
# #                            word_intrusion_features[indices_of_instances, :]
# #                            
# #                        responses[k*num_instances_per_topic + i] = \
# #                            word_topic[indices_of_instances, k]
# #                    break
# #                else:
# #                    continue
			
# 	data_frame = np.column_stack((instances, responses))
# 	np.random.shuffle(data_frame)
# 	num_train = np.floor(0.8 * num_instances)
# 	train_set = data_frame[0:num_train, :]
# 	test_set = data_frame[num_train:, :]
	
# 	svr = svm.SVR(C = 1.0, epsilon = 0.2)
# 	svr.fit(train_set[:, :-1], train_set[:, -1])
# 	predict_train = svr.predict(train_set[:, :-1])
# 	predict_test = svr.predict(test_set[:, :-1])
# 	# predict test is a list of predicted word topic probability
	


# 	index = list(predict_test).index(min(list(predict_test)))
# 	index_true = list(test_set[:, -1]).index(min(list(test_set[:, -1])))



	
















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

def evaluation_classification(source_file=DEFAULT_SOURCE_FILE, result_file = DEFAULT_RESULT_FILE):
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
		train_loose_accuracy, test_loose_accuracy, NMI = classification(data.doc_term_matrix, data._doc_label, data._label_category, data_percentage)
		res_file.write('%f\t%f' % (train_loose_accuracy, test_loose_accuracy))
	print 'The NMI between topic model and true label is\t' + str(NMI)
	return NMI

def evaluation_word_intrusion(doc_path, stop_word_path, lemmatize = True, Stem = False, source_file = DEFAULT_SOURCE_FILE):
	doc_term_matrix, number_of_doc, wordlist = preprocessing(doc_path, stop_word_path, lemmatize = lemmatize, Stem = Stem)
	data_file = str(source_file)
	with open(data_file, 'r') as INFILE:
		data = pickle.load(INFILE)
	word_topic = data._topic_word
	accuracy = WordIntrusion(doc_term_matrix, 10, len(wordlist), word_topic)

def evaluation_clustering(source_file=DEFAULT_SOURCE_FILE, input_file = 'adjacentMatrixUnderCS_10000'):
	data_file = str(source_file)
	with open(data_file, 'r') as INFILE:
		data = pickle.load(INFILE)

	NMI_network = networkclustering.networkClustering(input_file, data)

	return NMI_network


if __name__ == "__main__":
	NMI_topic_network = evaluation_classification()
	NMI_network = evaluation_clustering()


