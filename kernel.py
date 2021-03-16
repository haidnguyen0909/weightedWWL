#import Levenshtein
from wl import *
import numpy as np
import ot
from sklearn.metrics.pairwise import laplacian_kernel, rbf_kernel
from smoothot.dual_solvers import solve_dual, solve_semi_dual
from smoothot.dual_solvers import NegEntropy, SquaredL2
from smoothot.dual_solvers import get_plan_from_dual, get_plan_from_semi_dual
import cvxopt
import scipy
from svmopt import _w_distance_FE, optimize_ws
from scipy.sparse import lil_matrix, kron, identity
from scipy.sparse.linalg import lsqr

ALPHA = 1.0

def preprocess(y):
	tmp = []
	vals, freqs = np.unique(y, return_counts=True)
	if len(vals) != 2:
		return np.array(y)
	else:
		for i in range(len(y)):
			if y[i] == vals[0]:
				tmp.append(1)
			else:
				tmp.append(-1)
	return np.array(tmp)

def combine_kernels(Ds, gamma):
	n = len(Ds[0])
	K = np.zeros((n,n))
	for D in Ds:
		K+= laplacian_kernel(D, gamma =gamma)
	return K/len(Ds)



'''
random walk kernel
'''
def norm(adj):
	norm = adj.sum(axis=0)
	norm[norm==0] = 1
	return adj/norm

def computeRandomWalk(g1, g2):
	lmb=0.5
	g1 = nx.adj_matrix(g1)
	g2 = nx.adj_matrix(g2)
	norm1 = norm(g1)
	norm2 = norm(g2)
	w_prod = kron(lil_matrix(norm1), lil_matrix(norm2))
	starting_prob = np.ones(w_prod.shape[0]) / (w_prod.shape[0])
	stop_prob = starting_prob
	A = identity(w_prod.shape[0]) - (w_prod * lmb)
	x = lsqr(A, starting_prob)
	res = stop_prob.T.dot(x[0])
	return res
def computeRandomWalkKernel(graphs):
	n = len(graphs)
	res = np.zeros((n,n))
	for i in range(n):
		for j in range(i+1, n):
			res[i,j] = computeRandomWalk(graphs[i], graphs[j])
			res[j,i] = res[i,j]
	return res
'''
finish random walk kernel
'''

'''
shortest path kennel'''
def get_max_path(graphs, unweight):
	i = 0
	maxi=0
	for g in graphs:
		adj = nx.adjacency_matrix(g)
		floyd = scipy.sparse.csgraph.floyd_warshall(adj, directed=False,
						return_predecessors=False, unweighted=unweight)
		maxi = max(maxi, (floyd[~np.isinf(floyd)]).max())
		i=i+1
	return int(maxi)

def compute_splom(maxpath, graphs, unweight):
	res = lil_matrix(np.zeros((len(graphs), maxpath+1)))
	i = 0
	maxi=0
	for i,g in enumerate(graphs):
		adj = nx.adjacency_matrix(g)
		#print(adj)
		floyd = scipy.sparse.csgraph.floyd_warshall(adj, directed=False,
			return_predecessors=False, unweighted=unweight)
		#print(floyd)
		maxi = max(maxi, (floyd[~np.isinf(floyd)]).max())
		subsetter = np.triu(~(np.isinf(floyd)))
		ind = floyd[subsetter]
		accum = np.zeros(maxpath + 1)
		#print(tmind.max())
		tmp = int(ind.max()+1)
		accum[:tmp] += np.bincount(ind.astype(int))
		accum=accum/sum(accum)
		res[i] = lil_matrix(accum)
		#print(i,accum)
	return res

def shortest_path_kernel_matrix(graphs):
	unweight=True
	maxpath = get_max_path(graphs, unweight)
	print("maxpath=",maxpath)
	splom = compute_splom(maxpath, graphs, unweight)
	res = np.asarray(splom.dot(splom.T).todense())
	#print(res)
	#exit(1)
	return res

def load_OA_kernels(name, num_iterations, n):
	kernels = []
	for it in range(1, num_iterations+1):
		filename = "./gram/S"+name+"__WLOA_" + str(it)+".gram"
		kernel = []

		with open(filename) as f:
			for line in f:
				if line[-1]=='\n':
					line = line[:-1]
				row = []
				eles = line.split(' ')
				for ind, ele in enumerate(eles):
					if ind == 0 or ind == 1:
						continue
					words = ele.split(':')
					row.append(float(words[1]))
				kernel.append(row)
		kernel = np.array(kernel)
		kernels.append(kernel[:n, :n])
	return kernels

def wl_embeddings(Gs, h):
	wl = WL()
	Gs_list = wl._fit_transform(Gs, h)
	id2labels = wl._inv_label_dicts
	n = len(Gs)
	
	label_sequences = [
		np.full((len(G.nodes),h+1),np.nan) for G in Gs]
	total = 0
	for it in range(0, h+1):
		#if it==0: 
		#	continue
		G_s = Gs_list[it]
		id2label = id2labels[it]
		total+= len(id2label)
		for i,G in enumerate(G_s):
			ids = [int(G.nodes[node]['label']) for node in G.nodes]
			label_sequences[i][:, it] = ids
	return label_sequences


def computeWLSubTreeKernels(Gs, num_iterations):
	WLSubtreeKs = []
	for it in range(1, num_iterations+1):
		WLSubtreeK = computeWLSubtreeKernel(Gs, it)
		WLSubtreeKs.append(WLSubtreeK)
	return WLSubtreeKs

def computeWLSubtreeKernel(Gs, h):
	n = len(Gs)
	wl = WL()
	id2labels = wl._inv_label_dicts
	K_sum = np.zeros((n,n))
	Gs_list = wl._fit_transform(Gs, h)
	count = 0
	for it in range(1, h+1):
		count+=1
		G_s = Gs_list[it]
		id2label = id2labels[it]
		d = len(id2label)
		K = np.zeros((n,n))
		for i, G_i in enumerate(G_s):
			for j, G_j in enumerate(G_s[i:]):
				ind_i =  [int(G_i.nodes[node]['label']) for node in G_i.nodes]
				ind_j =  [int(G_j.nodes[node]['label']) for node in G_j.nodes]
				ind_i, xi = np.unique(ind_i, return_counts=True)
				ind_j, xj = np.unique(ind_j, return_counts=True)
				xi = xi/np.sum(xi)
				xj = xj/np.sum(xj)
				yi = np.zeros(d)
				yj = np.zeros(d)
				yi[ind_i] = xi
				yj[ind_j] = xj
				K[i, i+j] = np.sum(yi * yj)
		K = K + K.T
		K_sum += K
	return K_sum/count

def computeWWLDistances(Gs, num_iterations):
	WWLDistances = []
	for it in range(0, num_iterations+1):
		WWLDistance = wasserstein_distance(Gs,  it)
		WWLDistances.append(WWLDistance)
	return WWLDistances

def wasserstein_distance(Gs, num_iterations):
	it = num_iterations
	label_sequences = wl_embeddings(Gs,it)
	n = len(label_sequences)
	WWLDistance = np.zeros((n,n))
	emb_size = label_sequences[0].shape[1]
	for i, Gi in enumerate(label_sequences):
		labels_1 = label_sequences[i]
		for j, Gj in enumerate(label_sequences[i:]):
			labels_2 = label_sequences[i + j]
			costs = ot.dist(labels_1, labels_2, metric='hamming')
			mat = ot.emd(np.ones(len(labels_1))/len(labels_1), np.ones(len(labels_2))/len(labels_2), costs)
			WWLDistance[i, i+j] = np.sum(np.multiply(mat, costs)) #+ ALPHA
	WWLDistance = WWLDistance + WWLDistance.T
	return WWLDistance

def learnOneEpoch(Gs, ys, num_iterations, full_batch = True):
	beta = 0.00001
	wl = WL()
	Gs_list = wl._fit_transform(Gs, num_iterations)
	id2labels = wl._inv_label_dicts
	Gs_list = wl._fit_transform(Gs, num_iterations)
	train_index = [i for i in range(len(Gs))]
	test_index = [i for i in range(len(Gs))]
	#print(train_index,"dkm")
	id2labels = wl._inv_label_dicts
	if full_batch:
		minibatch = len(Gs)
	else:
		minibatch=2
	ws, shrunk_list = optimize_ws(Gs_list, ys, id2labels, train_index, test_index, num_iterations, beta, n_epoch=1, minibatch=minibatch)


def LearnParametricDistance(Gs, ys, num_iterations, is_weight, train_index, test_index):
	n = len(Gs)
	wl = WL()
	Gs_list = wl._fit_transform(Gs, num_iterations)
	id2labels = wl._inv_label_dicts
	ws = []
	beta = 0.001
	distances =[]
	B = len(train_index)
	for n_iters in range(1, num_iterations+1):
		D = np.zeros((n,n))
		print("optimizing for iter:", n_iters)
		if is_weight:
			ws, shrunk_list = optimize_ws(Gs_list, ys, id2labels, train_index, test_index, n_iters, beta, n_epoch=500, minibatch=2)
		else:
			for it in range(0, num_iterations+1):
				ws.append(np.ones(len(id2labels[it])))
		for i in range(n):
			for j in range(i, n):
				dist = 0.0
				for it in range(0, n_iters + 1):
					id2label = id2labels[it]
					Gi = Gs_list[it][i]
					ind_i = [int(Gi.nodes[node]['label']) for node in Gi.nodes]
					ind_i, xi = np.unique(ind_i, return_counts=True)
					xi = xi/np.sum(xi)

					Gj = Gs_list[it][j]
					ind_j = [int(Gj.nodes[node]['label']) for node in Gj.nodes]
					ind_j, xj = np.unique(ind_j, return_counts=True)
					xj = xj/np.sum(xj)

					ind, feat = _w_distance_FE(np.ones(len(id2label)), ind_i, xi, ind_j, xj, id2label)
					dist_it = np.dot(ws[it][ind], feat) 
					dist += dist_it
				distance = 1 - 1.0/(n_iters+1) * dist
				#if i == j:
				#	D[i, j] = distance + ALPHA*0.5
				#else:
				#	D[i, j] = distance + ALPHA
				#print(i, j, distance, dist)
				D[i,j] = distance
		D = D + D.T
		distances.append(D)
	return distances
