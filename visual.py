import matplotlib
import matplotlib.pyplot as plt
import random
import pickle
import networkx as nx
from collections import defaultdict


def get_random_color(pastel_factor = 0.5):
	return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
	return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
	max_distance = None
	best_color = None
	for i in range(0,100):
		color = get_random_color(pastel_factor = pastel_factor)
		if not existing_colors:
			return color
		best_distance = min([color_distance(color,c) for c in existing_colors])
		if not max_distance or best_distance > max_distance:
			max_distance = best_distance
			best_color = color
	return best_color


def visualize(savefile, i):
	adj = pickle.load(open('PROCESSED/adjacentMatrixUnderCS_1000', 'rb'))
	adj = nx.from_scipy_sparse_matrix(adj)

	dic = defaultdict(list)
	with open('PROCESSED/PaperToKeywords_1000.txt', 'r') as outfile:
		pid = 0
		for line in outfile:
			keywords = line.strip().split('\t')
			keyword = keywords[random.randint(0, len(keywords)-1)]
			dic[keyword].append(pid)
			pid += 1

	plt.figure(i)
	# pos = nx.spectral_layout(adj)
	pos = nx.fruchterman_reingold_layout(adj)
	# pos = nx.spring_layout(adj)

	color = list()

	for i in range(0,10):
		color.append(generate_new_color(color, pastel_factor = 0.9))
	
	i = 0
	for keywords, pid_list in dic.items():
		nx.draw_networkx_nodes(adj, pos, nodelist = pid_list, node_color = color[i], node_size = 10, alpha = 0.7)
		i += 1
	nx.draw_networkx_edges(adj, pos, width=1.0, alpha=0.5)
	plt.title('Network Visualization for 1000 documents')
	plt.savefig(save)

if __name__ == '__main__':
	for i in range(10):
		save = './visualize__' + str(i) + '.png'
		visualize(save, i)