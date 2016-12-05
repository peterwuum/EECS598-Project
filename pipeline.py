import os
import numpy as np
if __name__ == '__main__':
  for lambda_par in np.arange(0.1, 0.7, 0.05):
    for gamma_par in np.arange(0.05, 0.3, 0.05):
      print 'start lambda %f and gamma %f' % (lambda_par, gamma_par)
      filename = 'EVALUATION/plsa_lambda%f_gamma%f' % (lambda_par, gamma_par)
      os.system('python netplsa_with_plsa.py -rf %s -l %f -g %f' % (filename, lambda_par, gamma_par))
      os.system('python evaluation.py -sf %s -rf %s_evaluation' % (filename, filename))