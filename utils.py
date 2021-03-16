import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re
import untangle


def load_data(ds_name, use_node_label=True, use_node_attributes=False):
	Gs = []
	node2graph = {}
	with open("./%s/%s_graph_indicator.txt" % (ds_name, ds_name), "r") as f:
		c = 1
		for line in f:
			#print(line[:-1])
			node2graph[c] = int(line[:-1])
			if not node2graph[c] == len(Gs):
				Gs.append(nx.Graph())
			Gs[-1].add_node(c)
			c = c + 1
	print("Finished loading graph indicator...")
	#print(node2graph)
	with open("./%s/%s_A.txt" % (ds_name, ds_name), "r") as f:
		for line in f:
			edge = line[:-1].split(",")
			edge[1] = edge[1].replace(" ","")
			Gs[node2graph[int(edge[0])]-1].add_edge(int(edge[0]), int(edge[1]), weight = 1)


	print("Finished loading graph adjacent matrix...")
	if use_node_label:
		d = {}
		with open("./%s/%s_node_labels.txt" % (ds_name, ds_name), "r") as f:
			c = 1
			for line in f:
				node_label = (line[:-1])
				#print(node_label)
				if node_label not in d:
					d[node_label] = len(d)
				Gs[node2graph[c]-1].nodes[c]['label'] = node_label
				#print(c, node_label)
				c+=1
		print("Finished loading node labels...")
	if use_node_attributes:
		with open("./%s/%s_node_attributes.txt" % (ds_name, ds_name), "r") as f:
			c = 1
			for line in f:
				node_attributes = line[:-1].split(',')
				node_attributes = [float(attribute) for attribute in node_attributes]
				Gs[node2graph[c]-1].nodes[c]['attributes'] = np.array(node_attributes)
				c+=1
	class_labels = []
	with open("./%s/%s_graph_labels.txt" % (ds_name, ds_name), "r") as f:
		for line in f:
			label = (line[:-1])
			class_labels.append(int(label))
	#class_labels = np.array(class_labels, dtype=np.float)
	print("Finished loading labels...")
	return Gs, class_labels





