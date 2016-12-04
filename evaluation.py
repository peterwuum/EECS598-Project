import numpy as np
from sklearn import svm
from collections import defaultdict

def classification(doc_topic, label_list, label_category, percentage, accuracy_measure = 'tight'):
	classifier_list = list()

	number_of_train = len(doc_topic) * percentage
	test_true_label = label_list[number_of_train : ]

	# doc -> list of label
	test_label = defaultdict(set)
	# label list is doc -> set of label
	for i in range(0, label_category):
		current_label = i
		train_label_list = list()
		for index in range(0, number_of_train):
			if current_label in label_list[index]:
				train_label_list.append(1)
			else:
				train_label_list.append(0)


		X_train = doc_topic[0 : number_of_train]
		X_test = doc_topic[number_of_train : ]
		
		# class_dict = dict()
		# class_dict[1] = np.bincount(train_label_list) / len(X_train)
		# model = svm.SVC(kernel='linear', class_weight = class_dict)
		
		model = svm.SVC(kernel='linear', class_weight = 'auto')
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
		print 'tight accuracy: ' + str(count)
		return count
	else:
		count = 0
		for i in range(0, len(test_label)):
			if len(test_label[i].intersection(test_true_label[i])) >= 1:
				count += 1
		print 'loose accuracy: ' + str(count)
		return count
