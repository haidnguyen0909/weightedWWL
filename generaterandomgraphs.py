import networkx as nx
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
	

	#nx.draw_networkx_labels(G,pos=nx.spring_layout(G))
	#nx.draw(G)
	#plt.draw()
	#plt.show()
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





#gnp_random_graph_neg(20, 0.2)










