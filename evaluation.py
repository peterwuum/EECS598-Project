import numpy as np
from sklearn import svm
from collections import defaultdict

def classification(doc_topic, label_list, label_category, percentage, accuracy_measure = 'tight'):
	classifier_list = list()
	# doc -> list of label
	test_label = defaultdict(set)
	# label list is doc -> set of label
	for i in range(0, label_category):
		current_label = i
		label_list = list()
		for doc_label in label_list:
			if current_label in doc_label:
				label_list.append(1)
			else:
				label_list.append(0)

		number_of_train = len(doc_topic) * percentage
		X_train = doc_topic[0 : number_of_train]
		X_test = doc_topic[number_of_train : ]
		y_train = label_list[0 : number_of_train]
		y_test = label_list[number_of_train : ]
		class_dict = dict()
		class_dict[1] = len(doc_topic) / (label_category * np.bincount(1))
		
		model = svm.SVC(kernel='linear', class_weight = class_dict)
		model.fit(X_train, y_train)
		predict = model.predict(X_test)
		
		for index in range(0, len(predict)):
			test_label[index].append(current_label)

	if accuracy_measure == 'tight':
		count = 0
		for i in range(0, len(test_label)):
			if test_label[i] = label_list[i]:
				count += 1
		print 'tight accuracy: ' + str(count)
		return count
	else:
		count = 0
		for i in range(0, len(test_label)):
			if len(test_label[i].intersection(label_list[i])) >= 1:
				count += 1
		print 'loose accuracy: ' + str(count)
		return count









