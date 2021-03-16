

from wl import *
import numpy as np
import ot
from sklearn.metrics.pairwise import laplacian_kernel, rbf_kernel
from smoothot.dual_solvers import solve_dual, solve_semi_dual
from smoothot.dual_solvers import NegEntropy, SquaredL2
from smoothot.dual_solvers import get_plan_from_dual, get_plan_from_semi_dual
import cvxopt
from scipy.sparse import lil_matrix

def _w_distance_FE(w, ind_i, xi, ind_j, xj, id2label):
	m = len(ind_i)
	n = len(ind_j)
	#m = len(id2label)
	total = 0.0
	invalid_set = []
	feat = np.zeros(m)
	for i in range(m):
		density_i = xi[i]
		density_j = 0.0
		for j in range(n):
			if ind_i[i] == ind_j[j]:
				density_j = xj[j]
				break
		feat[i] = min(density_i,density_j)
	return ind_i, feat

def new_hinge_loss(ws, shrunk_list, Gs_list, ys, B1, id2labels, beta, beta_list,num_iterations, training=True):
	n = len(ys)
	total_loss = 0.0
	B2 = B1 - 10.0
	if training:
		mark = 1.0
	else:
		mark = 0.0
	count = 0
	correct = 0
	incorrect = 0
	for i in range(n):
		for j in range(i+1, n):
			dist = 0.0
			count +=1
			for it in range(0, num_iterations + 1):
				#if it ==0:
				#	continue
				id2label = id2labels[it]
				Gi = Gs_list[i][it]
				ind_i = [int(Gi.nodes[node]['label']) for node in Gi.nodes]
				ind_i, xi = np.unique(ind_i, return_counts=True)
				
				#print(ind_i, xi, np.sum(xi), xi/np.sum(xi))
				xi = xi/np.sum(xi)

				Gj = Gs_list[j][it]
				ind_j = [int(Gj.nodes[node]['label']) for node in Gj.nodes]
				ind_j, xj = np.unique(ind_j, return_counts=True)
				xj = xj/np.sum(xj)

				


				#St, Dt = new_node_distance(ind_i, ind_j, id2label)
				ind_i, feat = _w_distance_FE(np.ones(len(id2label)), ind_i, xi, ind_j, xj, id2label)
				#dist_it_tmp = np.dot(ws[it], zero)

				dist_it = np.dot(ws[it][ind_i], feat)
				#print(St, Dt)
				#print("dkm", dist_it_tmp, dist_it)

				dist += dist_it
			dist = 1.0 - 1.0/(num_iterations+1) * dist 
			#dist = min(max(dist, 0.0),1.0)
			if ys[i]*ys[j] == 1:
				loss = max(0, dist - B2)
			else:
				loss = max(0, B1 - dist)
			reg = 0.0
			if loss == 0.0:
				correct+=1
			else:
				incorrect+=1
			for it in range(0, num_iterations + 1):
				#if it !=1:
				#	continue
				beta_ = beta_list[it]
				shrunk = shrunk_list[it]
				reg += beta_ * np.sum((ws[it][shrunk])*(ws[it][shrunk])) * mark
			total_loss += loss + reg
	return total_loss, correct, incorrect


def Update(ws, shrunk_list, B1, grads, grad_B1, lr, beta, beta_list, num_iterations, inds):
	b = len(grads)
	for grad, ind in zip(grads, inds):
		if len(grad) != 0:
			for it in range(1, num_iterations+1):
				i = ind[it]
				ws[it][i] = ws[it][i] - lr/b * (grad[it]) 
	for it in range(1, num_iterations+1):
		beta_=beta_list[it]
		shrunk = shrunk_list[it]
		#ws[it][shrunk] = ws[it][shrunk]- lr * 2 * beta_ * (ws[it][shrunk]-1)
		ws[it][shrunk]= ws[it][shrunk].clip(min=0.0, max=2.0)
		#print(it, ws[it][shrunk])
	return ws, B1

def HingeGrad(ws, Gs_list, ys, B1, id2labels, beta, beta_list,num_iterations):
	n = len(ys)
	B2 = B1 - 10
	total_loss = 0.0
	grads = []
	B1_grad = 0.0
	total_grads = []
	total_inds = []
	count = 0

	for i in range(n):
		for j in range(i+1, n):
			dist = 0.0
			tmps_feat = []
			tmps_ind = []
			
			for it in range(0, num_iterations + 1):
				id2label = id2labels[it]
				Gi = Gs_list[i][it]
				ind_i = [int(Gi.nodes[node]['label']) for node in Gi.nodes]
				ind_i, xi = np.unique(ind_i, return_counts=True)
				xi = xi/np.sum(xi)

				Gj = Gs_list[j][it]
				ind_j = [int(Gj.nodes[node]['label']) for node in Gj.nodes]
				ind_j, xj = np.unique(ind_j, return_counts = True)
				xj = xj/np.sum(xj)
				ind, feat = _w_distance_FE(np.ones(len(id2label)), ind_i, xi, ind_j, xj, id2label)
				
				tmps_feat.append(- 1.0/(num_iterations+1)*feat)
				tmps_ind.append(ind)
				# skip level 0: atoms
				if it == 0:
					continue
				dist_it = np.dot(ws[it][ind], feat)
				dist += dist_it

			dist = 1.0 - 1.0/(num_iterations+1) * dist
			yi = ys[i]
			yj = ys[j]
			if yi == yj:
				loss = max(0, dist - B2) 
			else:
				loss = max(0, B1 - dist)
			reg = 0.0
			for it in range(1, num_iterations+1):
				beta_ = beta_list[it]
				reg += beta_ * np.sum((ws[it]-1)*(ws[it]-1))
			loss = loss + reg
			total_loss += loss

			grads = []

			for it in range(0, num_iterations+1):
				if yi==yj:
					if dist - B2 > 0.0:
						ind = tmps_ind[it]
						feat = tmps_feat[it]
						grads.append(feat)
						B1_grad -= 1.0
						count += 1
				else:
					if B1 - dist > 0.0:
						ind = tmps_ind[it]
						feat = tmps_feat[it]
						grads.append(-feat)
						B1_grad += 1.0
						count += 1
			total_grads.append(grads)
			total_inds.append(tmps_ind)
	if count == 0:
		return [], 0, 0.0, []
	return total_grads, B1_grad, total_loss, total_inds

def diff(a, b):
	dif = [i for i in a + b if i not in a or i not in b]
	return dif

def remove(Gs_list, ys, id2label, level):
	freqs = {}
	scores={}
	for i, Gs in enumerate(Gs_list):
		y = ys[i]
		G = Gs[level]
		inds = [int(G.nodes[node]['label']) for node in G.nodes]
		inds, x = np.unique(inds, return_counts=True)
		for i,ind in enumerate(inds):
			if ind not in freqs:
				freqs[ind] = 1
				scores[ind] = y
			else:
				freqs[ind]+=1
				scores[ind] += y*x[i]
	shrunk_list = []
	scores_={}
	for key, item in id2label.items():
		if key not in freqs.keys():
			#print(key, freqs.keys())
			continue
		if freqs[key] > 1:
			shrunk_list.append(key)
			scores_[key] = scores[key]
	return shrunk_list, scores_#np.array(shrunk_list)


def optimize_ws(Gs_list, ys, id2labels, train_index, test_index, num_iterations, beta, n_epoch, minibatch=2):
	ws = []
	for it in range(0, num_iterations+1):
		ws.append(np.ones(len(id2labels[it])))

	train_index = np.array(train_index)
	test_index = np.array(test_index)
	y_train = ys[train_index]
	y_test = ys[test_index]
	Gs_train = []
	Gs_test = []
	for i in train_index:
		Gs_train_it = []
		for it in range(0, num_iterations + 1):
			Gs_train_it.append(Gs_list[it][i])
		Gs_train.append(Gs_train_it)
	for i in test_index:
		Gs_test_it = []
		for it in range(0, num_iterations + 1):
			Gs_test_it.append(Gs_list[it][i])
		Gs_test.append(Gs_test_it)

	shrunk_list=[]
	scores_list=[]
	for it in range(0, num_iterations + 1):
		id2label = id2labels[it]
		new_list, scores = remove(Gs_train+Gs_test, np.concatenate((y_train, y_test)),id2label, it)
		shrunk_list.append(new_list)
		scores_list.append(scores)
	complement_list=[]
	for it in range(0, num_iterations + 1):
		id2label = id2labels[it]
		shrunk = shrunk_list[it]
		entire = [i for i in range(len(id2label))]
		complement = diff(entire, shrunk)
		complement_list.append(complement)
		#print(it, len(entire), len(shrunk), len(complement))
	beta_list=[]
	for it in range(0, num_iterations+1):
		shrunk = shrunk_list[it]
		beta_list.append(beta)# * 1.0/len(shrunk))

	#lr = 0.02
	lr = 0.01
	B1 = 5
	n_epoch = n_epoch

	for e in range(n_epoch):
		#print(ws)
		coupled = list(zip(Gs_train, y_train))
		np.random.shuffle(coupled)
		G_shuff, y_shuff = zip(*coupled)
		y_shuff = np.asarray(y_shuff)
		total_loss = 0.0
		grad_B1_total = 0.0
		count = 0.0
		check = 0.0
		for i in range(0, len(Gs_train), minibatch):
			count +=1
			tmp = min(len(Gs_train), i + minibatch)
			G_b = G_shuff[i:tmp]
			y_b = y_shuff[i:tmp]
			grad_w_b, grad_B1_b, loss_b, inds = HingeGrad(ws, G_b, y_b, B1, id2labels, beta, beta_list, num_iterations)
			grad_B1_total += grad_B1_b
			ws, B1 = Update(ws, shrunk_list, B1, grad_w_b, grad_B1_b, lr, beta, beta_list ,num_iterations, inds)
			#print(ws)
			total_loss += loss_b
		#print(e, ws)

	return ws, shrunk_list





