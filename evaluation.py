import numpy as np
from sklearn import svm
from collections import defaultdict
import pickle
import statistics
from netplsa_with_plsa import PLSA
from random import shuffle
from sklearn.model_selection import KFold

def classification(doc_topic, label_list, label_category_list, percentage, accuracy_measure = 'both'):
	label_list = np.array(label_list)
	classifier_list = list()
	doc_index = np.arange(len(label_list))
	kf = KFold(n_splits = 5, shuffle = True)
	train_loose_accuracy = []
	test_loose_accuracy = []
	for train, test in kf.split(doc_index):
		train_true_label = label_list[train]
		test_true_label = label_list[test]

		# doc -> list of label
		train_label = defaultdict(set)
		test_label = defaultdict(set)
		# label list is doc -> set of label
		for i in label_category_list:
			current_label = i
			train_label_list = list()
			for index in train:
				# print current_label
				# print label_list[index]
				if current_label in label_list[index]:
					train_label_list.append(1)
				else:
					train_label_list.append(0)


			X_train = doc_topic[train]
			X_test = doc_topic[test]
		
			model = svm.SVC(kernel='rbf', class_weight = 'balanced')
			# print train_label_list
			model.fit(X_train, train_label_list)
			predict_train = model.predict(X_train)

			for index in range(0, len(predict_train)):
				if predict_train[index] == 1:
					train_label[index].add(current_label)

			predict = model.predict(X_test)
		
			for index in range(0, len(predict)):
				if predict[index] == 1:
					test_label[index].add(current_label)

		if accuracy_measure == 'tight':
			count = 0
			for i in range(0, len(test_label)):
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
			for i in range(0, len(test_label)):
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
	print 'Mean train accuracy: ' + str(statistics.mean(train_loose_accuracy))
	print 'Mean test accuracy: ' + str(statistics.mean(test_loose_accuracy))
	return (statistics.mean(train_loose_accuracy), statistics.mean(test_loose_accuracy))


DEFAULT_SOURCE_FILE = "plsa_data"

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--source_file", "-sf", "source_file",
	default=DEFAULT_SOURCE_FILE,
	help="The path of source file")

def main(source_file=DEFAULT_SOURCE_FILE):
	data_file = source_file
	with open(data_file, 'r') as INFILE:
		data = pickle.load(INFILE)
	data_percentage = [0.8]
	for i in data_percentage:
		print 'data_percentage: ' + str(i)
		classification(data.doc_term_matrix, data._doc_label, data._label_category, i)
	

if __name__ == "__main__":
	main()

