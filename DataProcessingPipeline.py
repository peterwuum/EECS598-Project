import os
from multiprocessing import Process

def fetch_papers(number_of_papers_per_field):
  print 'start fetch %d papers per field' % number_of_papers_per_field
  os.system('python DataProcessing.py -o get_adj_matrix -nppf %d > out_%d' % (number_of_papers_per_field, number_of_papers_per_field))

if __name__ == "__main__":
  multi_process = False  

  for number_of_papers_per_field in range(1500, 10001, 500):
    # fetch_papers(number_of_papers_per_field)
    if multi_process:
      p = Process(target=fetch_papers, args=(number_of_papers_per_field,))
      p.start()
    else:
      fetch_papers(number_of_papers_per_field)
  print('Create all processes and wait.')
  

