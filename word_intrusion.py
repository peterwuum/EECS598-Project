def WordIntrusion(num_topwords = 5, intruder_ind = 200):
    #unigram_freq = self.doc_term_matrix.sum(axis = 0)
    #unigram_prob = unigram_freq / np.sum(self.doc_term_matrix)
    binary_doc_term = self.doc_term_matrix
    binary_doc_term[doc_term > 0] = 1
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

    for k in self.number_of_topic:
        ind_topwords_for_topic = np.argpartition(self._word_topic[:, k], -num_topwords)[-num_topwords:]
        #intruder_for_topic = self._word_topic[intruder_ind, k]
        #topwords = self._word_topic[:, k][ind]
        #doc_term_subset = self.doc_term_matrix[:, ind_topwords]
        topwords_feature = word_intrusion_features[ind_topwords_for_topic, :]
        intruder_feature = word_intrusion_features[intruder_ind, ]
        svr = svm.SVR(C=1.0, epsilon=0.2)
        svr.fit(topwords_feature, self._word_topic[ind_topwords_for_topic, k])
        predict_intrusion = svr.predict(topwords_feature)
