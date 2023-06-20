import networkx as nx
import os
import pickle
import math
import itertools
import matplotlib.pyplot as plt
from networkx.utils import py_random_state
import random

@py_random_state(2)
def gnp_random_graph_neg(n, p, seed=None):
	G = nx.Graph()
	labels = ['A', 'B', 'C']
	#G.add_nodes_from(range(n))
	for i in range(n):
		lab = random.choice(labels)
		G.add_node(i, label=lab)

	for i in range(n):
		for j in range(i+1, n):
			number = random.uniform(0,1)
			if number < p:
				G.add_edge(i, j)
	#nx.draw(G)
	#plt.draw()
	#plt.show()

	return G

def gnp_random_graph_pos(n, p, seed=None):
	#edges = itertools.combinations(range(10,n), 2)
	labels = ['A', 'B', 'C']
	G=nx.Graph()
	'''
	add edge

	'''
	G.add_edge(0,1)
	G.add_edge(0,2)
	G.add_edge(0,3)
	G.add_edge(1,4)
	G.add_edge(1,5)
	G.add_edge(2,6)
	G.add_edge(2,7)
	G.add_edge(3,8)
	G.add_edge(3,9)

	#G.add_nodes_from(range(10))
	G.add_node(0, label=labels[0])
	G.add_node(1, label=labels[0])
	G.add_node(2, label=labels[1])
	G.add_node(3, label=labels[2])
	G.add_node(4, label=labels[0])
	G.add_node(5, label=labels[1])
	G.add_node(6, label=labels[0])
	G.add_node(7, label=labels[0])
	G.add_node(8, label=labels[0])
	G.add_node(9, label=labels[1])
	for i in range(10, n):
		lab = random.choice(labels)
		G.add_node(i, label=lab)
	#G.add_nodes_from(range(10,n))
	#G.add_edges_from(edges)
	for i in range(10):
		for j in range(10,n):
			number = random.uniform(0,1)
			if number < p:
				G.add_edge(i,j)
	for i in range(10,n):
		for j in range(i+1, n):
			number = random.uniform(0,1)
			if number < p:
				G.add_edge(i, j)


	# nx.draw_networkx_labels(G,pos=nx.spring_layout(G))
	# nx.draw(G)
	# plt.draw()
	# plt.show()
	return G


def simulationdata(nGs, n, p):
	Gs= []
	#nGs=[]
	labels = []
	for i in range(nGs):
		Gs.append(gnp_random_graph_pos(n, p))
		labels.append(1)
	for i in range(nGs):
		Gs.append(gnp_random_graph_neg(n,p))
		labels.append(-1)
	return Gs, labels


def load_hc3_dataset(domain, folder_path='/Users/zaemyungkim/Development/human_vs_machine_texts/discourse_parsed'):
	def load_graphs(domain):
		path = os.path.join(folder_path, f'graphs_for_{domain}.pkl')
		with open(path, 'rb') as f:
			return pickle.load(f)

	# hc3_graphs = {
    #     'finance': load_graphs('finance'),
    #     'medicine': load_graphs('medicine'),
    #     'open_qa': load_graphs('open_qa'),
    #     'reddit_eli5': load_graphs('reddit_eli5'),
    #     'wiki_csai': load_graphs('wiki_csai')
    # }

	hc3_graphs = load_graphs(domain)
	# print(hc3_graphs['chatgpt'][0])
	Gs = []
	labels = []
	for chatgpt, human in zip(hc3_graphs['chatgpt'], hc3_graphs['human']):
		assert chatgpt['id'] == human['id']
		chatgpt_graph = chatgpt['graph']
		human_graph = human['graph']
		for node in chatgpt_graph.nodes():
			chatgpt_graph.nodes[node]['label'] = chatgpt_graph.nodes[node]['label_0']
		for node in human_graph.nodes():
			human_graph.nodes[node]['label'] = human_graph.nodes[node]['label_0']
		Gs.append(chatgpt_graph)
		labels.append(-1)
		Gs.append(human_graph)
		labels.append(1)

	return Gs, labels

#gnp_random_graph_neg(20, 0.2)










