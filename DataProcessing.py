import pickle
import numpy as np
import scipy as sp
from scipy.sparse import csc_matrix
import click, os, random

from collections import deque
from Queue import PriorityQueue


"""
Instruction:

optional input:
  the file which contain 10 area we choose

  1. get adjacent matrix, command: python fetchCitationNetwork.py -o get_adj_matrix 


"""



# Initialization
number_of_papers = 100
DEFAULT_OPERATION = ""
DEFAULT_CHOSEN_FIELD_FILE = "filtered_10_fields.txt"

def parse_fields(chosen_fields_file):
  keywords = set()
  with open(chosen_fields_file, 'r') as data:
    for line in data:
      keywords.add(line.strip().split('\t')[0])
  return keywords

def get_paper_id_under_field(keywords):
  paper_to_keywords = {}
  keyword_to_papers = {}
  paper_ids = set()
  # paperKeywordsUnderCS = open('PROCESSED/PaperKeywordsUnderCS', 'w')
  with open('DATA/PaperKeywords.txt', 'r') as data:
    for line in data:
      tmp = line.strip().split('\t')
      paper_id = tmp[0]
      keyword_text = tmp[1]
      keyword = tmp[2]
      if keyword in keywords:
        if keyword not in keyword_to_papers:
          keyword_to_papers[keyword] = set()
        if paper_id not in paper_to_keywords:
          paper_to_keywords[paper_id] = set()
        keyword_to_papers[keyword].add(paper_id)
        paper_to_keywords[paper_id].add(keyword)
        # paperKeywordsUnderCS.write("%s\t%s\t%s\n" % (paper_id, keyword_text, keyword))
        paper_ids.add(paper_id)

  return (paper_ids, paper_to_keywords, keyword_to_papers)

def create_graph_based_on_paper_ids(paper_ids):
  graph = {}
  with open('DATA/PaperReferences.txt', 'r') as data:
    for line in data:
      paper1, paper2 = line.strip().split('\t')
      if paper1 in paper_ids and paper2 in paper_ids:
        if paper1 not in graph:
          graph[paper1] = set()
        if paper2 not in graph:
          graph[paper2] = set()
        
        graph[paper1].add(paper2)
        graph[paper2].add(paper1)

  return graph

def get_seeds_from_each_keyword(graph, keyword_to_papers):
  seed_based_on_area = {}
  for area, papers in keyword_to_papers.items():
    max_degree_papers = []
    for paper in papers:
      if paper not in graph:
        continue
      degree = len(graph[paper])
      max_degree_papers.append((degree, paper))
      if len(max_degree_papers) >= number_of_papers:
        max_degree_papers = sorted(max_degree_papers, key=lambda x: -x[0])[:number_of_papers]
    seed_based_on_area[area] = max_degree_papers

  return seed_based_on_area

def bfs(paper_id, area, paper_to_keywords, graph, papers_based_on_area, keywords_under_CS, visited):
  queue = deque()
  queue.append(paper_id)

  while len(queue) > 0:
    curr = queue.popleft()

    if len(papers_based_on_area[area]) >= number_of_papers:
      break
    elif curr not in visited and curr in paper_to_keywords:
      for keyword in paper_to_keywords[curr]:
        if keyword in keywords_under_CS and len(papers_based_on_area[keyword]) < number_of_papers:
          papers_based_on_area[keyword].add(curr)

          # Add the neighbors into queue
          cnt = 0
          if curr not in graph:
            continue
          for neighbor in graph[curr]:
            if neighbor not in visited and cnt < 10:
              queue.append(neighbor)
              cnt += 1
          break

    visited.add(curr)
  

def get_1000_connected_papers(fields_file):
  keywords = parse_fields(fields_file)
  paper_ids, paper_to_keywords, keyword_to_papers = get_paper_id_under_field(keywords)
  graph = create_graph_based_on_paper_ids(paper_ids)
  seed_based_on_area = get_seeds_from_each_keyword(graph, keyword_to_papers)

  print ('start get_1000_connected_papers()...')
  papers_based_on_area = {}
  # initialize paper_to_keywords
  for area, paper in seed_based_on_area.items():
    papers_based_on_area[area] = set()
  
  visited = set()
  for area, papers in seed_based_on_area.items():
    print ('start BFS for %s...' % area, len(papers))
    for degree, paper_id in papers:
      print ('start at area %s paper_id %s with degree %d' % (area, paper_id, degree))
      bfs(paper_id, area, paper_to_keywords, graph, papers_based_on_area, keywords, visited)
      if len(papers_based_on_area[area]) >= number_of_papers:
        break

  pickle.dump(papers_based_on_area, open('PROCESSED/connected_papers', 'wb'))
  chosen_paper_ids = []
  for area, papers in papers_based_on_area.items():
    for paper in papers:
      chosen_paper_ids.append(paper)

  create_adjacent_matrix(chosen_paper_ids)

  # TODO: 
  # 1. paper_to_keywords
  # 2. paperidsUnderCS
  # 3. titleUnderCS
  create_paper_id_list(chosen_paper_ids)
  create_title_list(chosen_paper_ids)
  create_paper_to_keywords(chosen_paper_ids, paper_to_keywords)

def create_paper_id_list(chosen_paper_ids):
  ids = open('PROCESSED/PaperIdsUnderCS.txt', 'w')  
  for paper_id in chosen_paper_ids:
    ids.write('%s\n' % (paper_id))
  ids.close()

def create_title_list(chosen_paper_ids):
  chosen_paper_ids_set = set(chosen_paper_ids)

  paper_to_title = {}
  titles = open('PROCESSED/titlesUnderCS.txt', 'w')
  with open('DATA/Papers.txt', 'r') as data:
    for line in data:
      tmp = line.strip().split('\t')
      paper_id = tmp[0]
      title = tmp[2]
      if paper_id in chosen_paper_ids_set:
        paper_to_title[paper_id] = title


  for paper_id in chosen_paper_ids:
    titles.write('%s\n' % (paper_to_title[paper_id]))
  titles.close()

def create_paper_to_keywords(chosen_paper_ids, paper_to_keywords):
  file = open('PROCESSED/PaperToKeywords.txt', 'w')
  for paper_id in chosen_paper_ids:
    keywords = ''
    for keyword in paper_to_keywords[paper_id]:
      keywords += keyword + '\t'
    keywords = keywords[:-1] + '\n'
    file.write(keywords)

def create_adjacent_matrix(chosen_paper_ids):
  id_count = 0
  ids_under_CS = {}
  #with open('paperIdsUnderCSLayer1Sampled1000Evenly.txt', 'r') as data:
  for paper_id in chosen_paper_ids:
    ids_under_CS[paper_id] = id_count
    id_count += 1
      

  num_of_document = len(ids_under_CS)
  adj_matrix = csc_matrix((num_of_document, num_of_document), dtype=np.float)

  with open('DATA/PaperReferences.txt', 'r') as data:
    for line in data:
      paper1, paper2 = line.strip().split('\t')
      if paper1 in ids_under_CS and paper2 in ids_under_CS:
        id1 = ids_under_CS[paper1]
        id2 = ids_under_CS[paper2]
        adj_matrix[id1, id2] = 1.0
        adj_matrix[id2, id1] = 1.0

  pickle.dump(adj_matrix, open('PROCESSED/adjacentMatrixUnderCS', 'wb'))



CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)

@click.option("--operation", "-o", "operation",
    default=DEFAULT_OPERATION,
    help="Operation you want to do")

@click.option("--fields_file", "-f", "fields_file",
    default=DEFAULT_CHOSEN_FIELD_FILE,
    help="The file that contains chosen fields")

def main(operation=DEFAULT_OPERATION, fields_file=DEFAULT_CHOSEN_FIELD_FILE):
  if operation == 'get_adj_matrix':
    get_1000_connected_papers(fields_file)
    

if __name__ == "__main__":
  main()
