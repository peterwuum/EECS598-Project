# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 20:36:09 2016

@author: XuefeiZhang
"""

import numpy as np
def overlapNMI(cluster1, cluster2):
    n,K = cluster1.shape
    A = np.dot((1-cluster1).T, (1-cluster2))
    B = np.dot((1-cluster1).T, cluster2)
    C = np.dot(cluster1.T, (1-cluster2))
    D = np.dot(cluster1.T, cluster2)
    Atemp = np.array(A)
    Atemp[A==0] = n
    Btemp = np.array(B)
    Btemp[B==0] = n
    Ctemp = np.array(C)
    Ctemp[C==0] = n
    Dtemp = np.array(D)
    Dtemp[D==0] = n
    BDtemp = np.array(B+D)
    BDtemp[B+D == 0] = n
    ACtemp = np.array(A+C)
    ACtemp[A+C == 0] = n
    ABtemp = np.array(A+B)
    ABtemp[A+B == 0] = n
    CDtemp = np.array(C+D)
    CDtemp[C+D == 0] = n
    sum1 = -(A*np.log2(Atemp/n) + B*np.log2(Btemp/n) + C*np.log2(Ctemp/n) + D*np.log2(Dtemp/n))
    sum2 = -((B+D)*np.log2(BDtemp/n) + (A+C)*np.log2(ACtemp/n))
    H_X_Y_mat = sum1-sum2
    
    #sum3 = -((A+B)*np.log2(ABtemp/n) + (C+D)*np.log2(CDtemp/n))
    
    #index = -(A*np.log2(Atemp/n) + D*np.log(Dtemp/n)) + (B*np.log2(Btemp/n) + C*np.log2(Ctemp/n))
    #H_star_X_Y = np.array(H_X_Y_mat)
    #H_star_X_Y[index < 0] = sum3[index < 0 ]
    
    X_marginal_1 = np.sum(cluster1, axis = 0)
    X_marginal_0 = np.sum(1-cluster1, axis = 0)
    Y_marginal_1 = np.sum(cluster2, axis = 0)
    Y_marginal_0 = np.sum(1-cluster2, axis = 0)
    
    X_marginal_1_temp = np.array(X_marginal_1)
    X_marginal_1_temp[X_marginal_1 == 0] = n
    X_marginal_0_temp = np.array(X_marginal_0)
    X_marginal_0_temp[X_marginal_0 == 0] = n
    
    Y_marginal_1_temp = np.array(Y_marginal_1)
    Y_marginal_1_temp[Y_marginal_1 == 0] = n
    Y_marginal_0_temp = np.array(Y_marginal_0)
    Y_marginal_0_temp[Y_marginal_0 == 0] = n


    H_X = np.min(H_X_Y_mat, axis = 0)
    H_X_marginal = X_marginal_1 * np.log2(X_marginal_1_temp/n) + \
    X_marginal_0 * np.log2(X_marginal_0_temp/n)
    H_X_Y_norm = H_X/H_X_marginal
    H_X_Y = np.sum(H_X_Y_norm)/K
    
    H_Y = np.min(H_X_Y_mat, axis = 1)

    #H_X_Y = sum(H_X)
    
    H_Y_X = sum(H_Y)
    H_Y_marginal = Y_marginal_1 * np.log2(Y_marginal_1_temp/n) + \
    Y_marginal_0 * np.log2(Y_marginal_0_temp/n)
    H_Y_X_norm = H_Y/H_Y_marginal
    H_Y_X = np.sum(H_Y_X_norm)/K

    NMI = 1 - 1/2(H_Y_X + H_X_Y)
 
    # H_X_marginal = -np.sum(X_marginal_1 * np.log2(X_marginal_1_temp/n) + \
    # X_marginal_0 * np.log2(X_marginal_0_temp/n))
    
    # H_Y_marginal = -np.sum(Y_marginal_1 * np.log2(Y_marginal_1_temp/n) + \
    # Y_marginal_0 * np.log2(Y_marginal_0_temp/n))

    
    NMI = 1 - 0.5*(H_X_Y/H_X_marginal + H_Y_X/H_Y_marginal)
    print 'The NMI of the overlapping community detection is' + str(NMI)
    return NMI
    
    
    
    