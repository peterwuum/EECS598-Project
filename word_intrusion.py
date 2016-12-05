import numpy as np
from sklearn import svm
def WordIntrusion(num_topwords = 10, intruder_ind = 200, num_instances_per_topic = 10):
    #unigram_freq = self.doc_term_matrix.sum(axis = 0)
    #unigram_prob = unigram_freq / np.sum(self.doc_term_matrix)
    binary_doc_term = self.doc_term_matrix
    binary_doc_term[binary_doc_term > 0] = 1
    bigram_freq = np.dot(binary_doc_term.T, binary_doc_term)
    unigram_prob = np.diag(bigram_freq) / np.trace(bigram_freq)
    #np.fill_diagonal(bigram_freq, 0)
    bigram_prob = bigram_freq / (np.sum(bigram_freq) - np.trace(bigram_freq))
    PMI_denominator = np.outer(unigram_prob, unigram_prob)
    PMI_mat = np.log(bigram_prob / PMI_denominator)
    PMI = PMI_mat.sum(axis=0)
    CP1 = (PMI_mat.dot(1.0 / unigram_prob)).sum(axis=0)
    CP2 = ((1.0 / unigram_prob).dot(PMI_mat)).sum(axis=0)
    word_intrusion_features = np.column_stack((PMI, CP1, CP2))
    
    ind_words = np.ndarray((self.number_of_topic, self._numWord))
    ind_topwords = np.ndarray((self.number_of_topic, num_topwords))
    num_instances = num_instances_per_topic * self.number_of_topic
    instances = np.ndarray((num_instances, 6))
    responses = np.zeros(num_instances)
    for k in range(self.number_of_topic):
        # get the indices that will sort the array, from index of the smallest number to that of the largest
        ind_words[k, :] = np.argsort(self._word_topic[:, k])
        # get the indices of the top 10 words in each topic
        ind_topwords[k, :] = ind_words[k, :][::-1][:num_topwords]
        
        #topwords_feature = word_intrusion_features[ind_topwords[k, :], :]
        #intruder_feature = word_intrusion_features[intruder_ind, ]
        
        
    for k in range(self.number_of_topic):
        #for i in range(num_instances_per_topic):
            # randomly sample 5 words from the top 10 words in each topic
        
        other_topwords_ind = np.delete(ind_topwords, k, axis=0).ravel()
        last_50%_ind = ind_words[k, :np.floor(0.5*self._numWord)]
        intersection = np.interset1d(other_topwords_ind, last_50%_ind)
        if intersection.size > 10:
            for i in range(num_instances_per_topic):
                topwords_ind = np.random.choice(ind_topwords[k, :], 0.5 * num_topwords)
                indices_of_instances = np.append(topwords_ind, intersection[i])
                instances[k*num_instances_per_topic + i, :] = \
                    word_intrusion_features[indices_of_instances, :]
                responses[k*num_instances_per_topic + i] = \
                    self._word_topic[indices_of_instances, k]
            
#            for j in range(self.number_of_topic):
#                if j != i:
#                    # Check if any top word in some other topic j rank in
#                    # the last 50% in the current topic i. If found, break the loop.
#                    intersection = np.intersect1d(ind_words[k, :np.floor(0.5*self._numWord)], 
#                                                  ind_topwords[j,:])
#                    if intersection.size != 0:
#                        indices_of_instances = np.append(topwords_ind, intersection[0])
#                        instances[k*num_instances_per_topic + i, :] = \
#                            word_intrusion_features[indices_of_instances, :]
#                            
#                        responses[k*num_instances_per_topic + i] = \
#                            self._word_topic[indices_of_instances, k]
#                    break
#                else:
#                    continue
            
    data_frame = np.column_stack((instances, responses))
    np.random.shuffle(data_frame)
    num_train = np.floor(0.8 * num_instances)
    train_set = data_frame[0:num_train, :]
    test_set = data_frame[num_train:, :]
    
    svr = svm.SVR(C=1.0, epsilon=0.2)
    svr.fit(train_set[:, :-1], train_set[:, -1])
    predict_train = svr.predict(train_set[:, :-1])
    predict_test = svr.predict(test_set[:, :-1])

    
