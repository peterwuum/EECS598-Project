import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def folder_to_adj_matrix(folder_path):
  adjs = []
  path = os.path.isdir(folder_path)
  if path:
    for f in os.listdir(folder_path):
      if os.path.isfile(os.path.join(folder_path, f)) and f.startswith('adjacentMatrixUnderCS_'):
        adjs.append(int(f.split('_')[1]))
  return adjs

def folder_to_avg_time(folder_path):
  time_files = []
  path = os.path.isdir(folder_path)
  if path:
    for f in os.listdir(folder_path):
      if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('_avg_runtime_in_seconds'):
        time_files.append(f)
  return time_files

def parse_num_of_papers_from_time_file(filename):
  return int(filename[:-1*len('_avg_runtime_in_seconds')].split('_')[-1])

def visualization():
  time_files = folder_to_avg_time('TIME')
  print 'time_files', time_files
  plsa_runtimes = {}
  netplsa_runtimes = {}
  for tf in time_files:
    with open('TIME/'+tf, 'r') as data:
      for line in data:
        plsa_runtime, netplsa_runtime = line.strip().split('\t')

      num_of_papers = parse_num_of_papers_from_time_file(tf)
      plsa_runtime = float(plsa_runtime)
      netplsa_runtime = float(netplsa_runtime)
      plsa_runtimes[num_of_papers] = plsa_runtime
      netplsa_runtimes[num_of_papers] = netplsa_runtime

  plsa_runtimes = sorted(plsa_runtimes.items(), key=lambda x: x[0])  
  netplsa_runtimes = sorted(netplsa_runtimes.items(), key=lambda x: x[0])  

  # plot plsa
  x = [tmp[0] for tmp in plsa_runtimes]
  y = [tmp[1] for tmp in plsa_runtimes]
  
  fig = plt.figure(1)
  ax = fig.add_subplot(111)
  ax.plot(x, y)
  ax.grid(True)
  ax.set_xlabel('Number of documents')
  ax.set_ylabel('Runtime in seconds')
  ax.set_title('PLSA runtime against number of documents')
  fig.savefig('TIME/plsa_runtime.png', transparent = True, bbox_inches='tight')

  # plot netplsa
  x = [tmp[0] for tmp in netplsa_runtimes]
  y = [tmp[1] for tmp in netplsa_runtimes]

  fig = plt.figure(2)
  ax = fig.add_subplot(111)
  ax.plot(x, y)
  ax.grid(True)
  ax.set_xlabel('Number of documents')
  ax.set_ylabel('Runtime in seconds')
  ax.set_title('NetPLSA runtime against number of documents')
  fig.savefig('TIME/netplsa_runtime.png', transparent = True, bbox_inches='tight')


"""
  Main function

"""

if __name__ == '__main__':
  lambda_par = 0.4
  gamma_par = 0.15
  
  adjs = folder_to_adj_matrix('PROCESSED')

  for num_of_papers in adjs:
  # for num_of_papers in range(1000, 4001, 1000):
    filename = 'TIME/plsa_data_%d' % (num_of_papers)
    os.system('python netplsa_with_plsa.py -dfs %d -rf %s -l %f -g %f' % (num_of_papers, filename, lambda_par, gamma_par))

  visualization()




