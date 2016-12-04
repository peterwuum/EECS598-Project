import numpy as np
from sklearn import svm
from collections import defaultdict
import pickle
from netplsa_with_plsa import PLSA
from random import shuffle

def classification(doc_topic, label_list, label_category_list, percentage, accuracy_measure = 'both'):
	classifier_list = list()
	number_of_train = int(len(doc_topic) * percentage)
	shuffle(label_list)

	test_true_label = label_list[number_of_train : ]

	# doc -> list of label
	test_label = defaultdict(set)
	# label list is doc -> set of label
	for i in label_category_list:
		current_label = i
		train_label_list = list()
		for index in range(0, number_of_train):
			# print current_label
			# print label_list[index]
			if current_label in label_list[index]:
				train_label_list.append(1)
			else:
				train_label_list.append(0)


		X_train = doc_topic[0 : number_of_train]
		X_test = doc_topic[number_of_train : ]
		
		# class_dict = dict()
		# class_dict[1] = np.bincount(train_label_list) / len(X_train)
		# model = svm.SVC(kernel='linear', class_weight = class_dict)
		
		model = svm.SVC(kernel='linear', class_weight = 'balanced')
		# print train_label_list
		model.fit(X_train, train_label_list)
		predict = model.predict(X_test)
		
		for index in range(0, len(predict)):
			if predict[index] == 1:
				test_label[index].add(current_label)

	if accuracy_measure == 'tight':
		count = 0
		for i in range(0, len(test_label)):
			if test_label[i] == test_true_label[i]:
				count += 1
		print 'tight accuracy: ' + str(float(count) / len(test_label))
		return float(count) / len(test_label)
	
	elif accuracy_measure == 'loose':
		count = 0
		for i in range(0, len(test_label)):
			if len(test_label[i].intersection(test_true_label[i])) >= 1:
				count += 1
		print 'loose accuracy: ' + str(float(count) / len(test_label))
		return float(count) / len(test_label)
	
	else:
		count_tight = 0
		count_loose = 0
		for i in range(0, len(test_label)):
			if test_label[i] == test_true_label[i]:
				count_tight += 1
			if len(test_label[i].intersection(test_true_label[i])) >= 1:
				count_loose += 1
		
		print 'tight accuracy: ' + str(float(count_tight) / len(test_label))
		print 'loose accuracy: ' + str(float(count_loose) / len(test_label))
		return (float(count_tight)/len(test_label), float(count_loose)/len(test_label))

if __name__ == '__main__':
	data_file = './plsa_data'
	with open(data_file, 'r') as INFILE:
		data = pickle.load(INFILE)
	data_percentage = [0.5, 0.6, 0.7, 0.8, 0.9]
	for i in data_percentage:
		print 'data_percentage: ' + str(i)
		classification(data.doc_term_matrix, data._doc_label, data._label_category, i)
