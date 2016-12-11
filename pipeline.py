import os
import numpy as np

def cross_validation_lambda_gamma():
  for lambda_par in np.arange(0.1, 0.7, 0.05):
    for gamma_par in np.arange(0.05, 0.3, 0.05):
      print 'start lambda %f and gamma %f' % (lambda_par, gamma_par)
      filename = 'EVALUATION/plsa_lambda%f_gamma%f' % (lambda_par, gamma_par)
      os.system('python netplsa_with_plsa.py -rf %s -l %f -g %f' % (filename, lambda_par, gamma_par))
      os.system('python evaluation.py -sf %s -rf %s_evaluation' % (filename, filename))
      os.system('rm %s' % (filename))

def cross_validation_synthetic_edges():
  lambda_par = 0.5
  gamma_par = 0.1
  for synthetic_edge_prob in np.arrange(0.05, 0.25, 0.05):
    filename = 'EVALUATION/plsa_lambda%f_gamma%f_syn%f' % (lambda_par, gamma_par, synthetic_edge_prob)
    os.system('python netplsa_with_plsa.py -rf %s -l %f -g %f -sep %f' % (filename, lambda_par, gamma_par, synthetic_edge_prob))
    os.system('python evaluation.py -sf %s -rf %s_evaluation' % (filename, filename))
    os.system('rm %s' % (filename))

if __name__ == '__main__':
  cross_validation_lambda_gamma()
  cross_validation_synthetic_edges()