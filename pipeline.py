import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def cross_validation_lambda_gamma():
  for lambda_par in np.arange(0.1, 0.7, 0.05):
    for gamma_par in np.arange(0.05, 0.3, 0.05):
      print 'start lambda %f and gamma %f' % (lambda_par, gamma_par)
      filename = 'EVALUATION/plsa_lambda%f_gamma%f' % (lambda_par, gamma_par)
      os.system('python netplsa_with_plsa.py -rf %s -l %f -g %f' % (filename, lambda_par, gamma_par))
      os.system('python evaluation.py -sf %s -rf %s_evaluation' % (filename, filename))
      os.system('rm %s' % (filename))

def cross_validation_synthetic_edges(range_prob):
  lambda_par = 0.25
  gamma_par = 0.25
  adj_matrix = 'EVALUATION/adjacentMatrixUnderCS_1000'
  for synthetic_edge_prob in range_prob:
    filename = 'EVALUATION/plsa_lambda%f_gamma%f_syn%f' % (lambda_par, gamma_par, synthetic_edge_prob)
    os.system('python netplsa_with_plsa.py -rf %s -l %f -g %f -sep %f' % (filename, lambda_par, gamma_par, synthetic_edge_prob))
    os.system('python evaluation.py -am %s -sf %s -rf %s_evaluation_classification > %s_evaluation_clustering' \
              % (adj_matrix, filename, filename, filename))
    os.system('rm %s' % (filename))

  return range_prob
    
def visualization_synthetic_edges(range_prob):
  netplsa = [0.279294490814, 0.267090153694, 0.265329885483, 0.299588871002, 0.247399067879, 0.26199362278, \
          0.288929772377, 0.260640239716, 0.269258069992, 0.272021889687, 0.284374594688, 0.305254530907, \
          0.270878052711, 0.29259622097, 0.280127811432]
  net = [0.291124606133, 0.232023477554, 0.228623628616, 0.132593870163, 0.131094158173, 0.128622817993, \
          0.162996196747, 0.128320121765, 0.14562253952, 0.171022987366, 0.195153093338, 0.170058584213, \
          0.106778669357, 0.140489673615, 0.0952262878418]

  fig = plt.figure(1)
  ax = fig.add_subplot(111)
  plt.plot(range_prob, netplsa, 'b', label="NetPLSA")
  plt.plot(range_prob, net, 'r', label="Spectral Clustering")
  plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., shadow=True, fancybox=True)

  ax.grid(True)
  ax.set_xlabel('Probability of synthetic connections')
  ax.set_ylabel('NMI')
  fig.savefig('EVALUATION/netplsa_clustering_NMI.png', transparent = True, bbox_inches='tight')


if __name__ == '__main__':
  # cross_validation_lambda_gamma()

  range_prob = np.arange(0.0, 0.176, 0.0125)
  # cross_validation_synthetic_edges(range_prob)
  visualization_synthetic_edges(range_prob)