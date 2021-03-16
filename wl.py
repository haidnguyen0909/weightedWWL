import numpy as np
import networkx as nx
import utils
import copy
import matplotlib.pyplot as plt
from utils import load_data
import sklearn
from sklearn import cluster
from sklearn.cluster import KMeans


def reverse_dic(dic):
	rdic = {}
	#print(dic)
	for key, value in zip(dic.keys(), dic.values()):
		#print(key, value)
		rdic[value] = key
	return rdic

class WL():
	def __init__(self):
		self._last_new_label = -1
		self._label_dict = {}
		self._label_dicts = {}
		self._inv_label_dicts = {}
		self._preprocess_relabel_dict = {}
		self._results = {}

	def _get_next_label(self):
		self._last_new_label +=1
		return self._last_new_label
	def _reset_label_generation(self):
		self._last_new_label = -1
	def _get_neighbor_label(self, G, sort = True):
		neighbor_indices = [[ne for ne in G.neighbors(node)] for node in G.nodes()]
		neighbor_labels_t = [[G.nodes[ne]['label'] for ne in G.neighbors(node)] for node in G.nodes()]
		neighbor_labels = []
		for n_labels in neighbor_labels_t:
			if sort:
				neighbor_labels.append(sorted(n_labels))
			else:
				neighbor_labels.append(n_labels)
		return neighbor_labels
	def _append_label_dict(self, merged_labels):
		for merged_label in merged_labels:
			dict_key = '-'.join(map(str, merged_label))
			if dict_key not in self._label_dict.keys():
				self._label_dict[dict_key] = self._get_next_label()
	def _relabel_graph(self, G, merged_labels):
		new_labels = []
		for merged in merged_labels:
			new_label = self._label_dict['-'.join(map(str, merged))]
			new_labels.append(new_label)
		return new_labels
	def _relabel_graphs(self, Gs):
		preprocessed_Gs = []
		for i,G in enumerate(Gs):
			preG = G.copy()
			labels = [G.nodes[node]['label'] for node in G.nodes()]
			new_labels = []
			for label in labels:
				if label in self._preprocess_relabel_dict.keys():
					new_label = self._preprocess_relabel_dict[label]
					new_labels.append(new_label)
				else:
					self._preprocess_relabel_dict[label] = self._get_next_label()
					new_label = self._preprocess_relabel_dict[label]
					new_labels.append(new_label)
			for node, new_label in zip(preG.nodes(), new_labels):
				preG.nodes[node]['label'] = new_label
			preprocessed_Gs.append(preG)

		self._reset_label_generation()
		return preprocessed_Gs

	def _fit_transform(self, Gs, n_iterations = 1):
		Gs_list = []
		Gs = self._relabel_graphs(Gs)
		Gs_c = copy.deepcopy(Gs)
		Gs_list.append(Gs_c)

		self._label_dicts[0] = copy.deepcopy(self._preprocess_relabel_dict)
		self._inv_label_dicts[0] = reverse_dic(self._label_dicts[0])

		
		for it in np.arange(0, n_iterations+1):
			self._reset_label_generation()
			self._label_dict = {}
			for i, G in enumerate(Gs):
				current_labels = [G.nodes[node]['label'] for node in G.nodes()]
				neighbor_labels = self._get_neighbor_label(G)
				merged_labels = [ [a] + b for a,b in zip(current_labels, neighbor_labels)]

				self._append_label_dict(merged_labels)
				new_labels = self._relabel_graph(G, merged_labels)
				for node, new_label in zip(G.nodes(), new_labels):
					G.nodes[node]['label'] = new_label
				#self._results[it][i] = (merged_labels, new_labels)
			Gs_c = copy.deepcopy(Gs)
			Gs_list.append(Gs_c)
			self._label_dicts[it] = copy.deepcopy(self._label_dict)
			self._inv_label_dicts[it] = reverse_dic(self._label_dicts[it])

		return Gs_list





