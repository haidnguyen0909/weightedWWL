import numpy as np
import pandas as pd
import argparse
import os
from sklearn.model_selection import ParameterGrid, StratifiedKFold, KFold
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,make_scorer, roc_auc_score
from kernel import *
from utils import *
from sklearn import manifold
from collections import Counter
from generaterandomgraphs import simulationdata
import time
import matplotlib.pyplot as plt

def grid_search_cv(model, param_grid, precomputed_kernels, y, cv=5):
	cv = StratifiedKFold(n_splits=cv, shuffle=False)
	results =[]
	for train_index, test_index in cv.split(precomputed_kernels[0], y):
		split_results = []
		params = []
		for idx, K in enumerate(precomputed_kernels):
			for p in list(ParameterGrid(param_grid)):
				sc = _fit_and_score(clone(model), K, y, scorer=make_scorer(accuracy_score),
					train=train_index, test=test_index, verbose=0, parameters=p,fit_params=None)
				split_results.append(sc)
				params.append({'K_idx':idx, 'params':p})
		results.append(split_results)
	results=np.array(results)
	fin_results = results.mean(axis=0)
	best_idx = np.argmax(fin_results)
	print(best_idx, fin_results[best_idx])
	ret_model = clone(model).set_params(**params[best_idx]['params'])
	return ret_model.fit(precomputed_kernels[params[best_idx]['K_idx']], y), params[best_idx]


def check(M1, M2, ys):
	n = len(M1)
	for i in range(n):
		for j in range(i+1, n):
			print(i, j, ys[i],ys[j],M1[i,j], M2[i, j])

#def run_cv_for_training(Gs, ,K0_s, K1_s, D_s, ys, num_iterations):
def run_cv_for_training(Gs, K0_s,ys, num_iterations):
	print("Running 10-cv...")
	cv = KFold(n_splits =10, shuffle=True)
	list_acc=[]
	index = 0
	for train_index, test_index in cv.split(K0_s[0], ys):
		index+=1
		D_s = LearnParametricDistance(Gs, ys, num_iterations, is_weight=False, train_index=train_index, test_index=test_index)
		Dw_s = LearnParametricDistance(Gs, ys, num_iterations, is_weight=True, train_index=train_index, test_index=test_index)
		#Dw_s = pairwise_distance(Gs, y, sinkhorn_lambda=0.01, kl_lambda=0.1, num_iterations = num_iterations, is_weight=True, train_index = train_index, test_index = test_index)
		##Dw_s = K0_s
		accs = run_train_test(K0_s, D_s, Dw_s, ys, train_index, test_index, num_iterations)
		list_acc.append(accs)
		print("term ", index, "  SVM ", accs)
		#return 0
	print(np.mean(list_acc, axis=0))


def run_train_test(K0_s, D_s ,Dw_s, ys, train_index, test_index, num_iterations):
	gammas = np.logspace(-5, -2,num=4)
	param_grid = [{'C': np.logspace(-3,3,num=7)}]
	kernel_matrices_0 = []
	kernel_matrices_1 = []
	kernel_matrices_2 = []
	kernel_matrices_3 = []
	kernel_params = []
	kernel_params_2 = []
	print("*****************")
	evals = []
	
	for gamma in gammas:
		for iter_ in range(0, num_iterations):
			K0 = K0_s[iter_]
			#K1 = K1_s[iter_]
			K2 = laplacian_kernel(Dw_s[iter_], gamma=gamma)
			K1 = laplacian_kernel(D_s[iter_], gamma=gamma)
			kernel_matrices_0.append(K0)
			kernel_matrices_1.append(K1)
			kernel_matrices_2.append(K2)
			kernel_params.append([gamma,iter_])
	
	y_train, y_test = ys[train_index], ys[test_index]

	

	K_train_0 = [K0[train_index][:, train_index] for K0 in kernel_matrices_0]
	K_test_0 = [K0[test_index][:, train_index] for K0 in kernel_matrices_0]
	gs_0, best_params_0 = grid_search_cv(SVC(kernel='precomputed'), param_grid, K_train_0, y_train, cv=5)
	C_0 = best_params_0['params']['C']
	gamma_0, iter_0 = kernel_params[best_params_0['K_idx']]
	
	K_train_1 = [K1[train_index][:, train_index] for K1 in kernel_matrices_1]
	K_test_1 = [K1[test_index][:, train_index] for K1 in kernel_matrices_1]
	gs_1, best_params_1 = grid_search_cv(SVC(kernel='precomputed'), param_grid, K_train_1, y_train, cv=5)
	C_1 = best_params_1['params']['C']
	gamma_1, iter_1 = kernel_params[best_params_1['K_idx']]
	

	K_train_2 = [K2[train_index][:, train_index] for K2 in kernel_matrices_2]
	K_test_2 = [K2[test_index][:, train_index] for K2 in kernel_matrices_2]
	gs_2, best_params_2 = grid_search_cv(SVC(kernel='precomputed'), param_grid, K_train_2, y_train, cv=5)
	C_2 = best_params_2['params']['C']
	gamma_2, iter_2 = kernel_params[best_params_2['K_idx']]


	y_pred_0 = gs_0.predict(K_test_0[best_params_0['K_idx']])
	y_pred_1 = gs_1.predict(K_test_1[best_params_1['K_idx']])
	y_pred_2 = gs_2.predict(K_test_2[best_params_2['K_idx']])

	y_pred_trn_0 = gs_0.predict(K_train_0[best_params_0['K_idx']])
	y_pred_trn_1 = gs_1.predict(K_train_1[best_params_1['K_idx']])
	y_pred_trn_2 = gs_2.predict(K_train_2[best_params_2['K_idx']])
	print("*********")
	acc0 = accuracy_score(y_test, y_pred_0)
	acc1 = accuracy_score(y_test, y_pred_1)
	acc2 = accuracy_score(y_test, y_pred_2)
	
	print(C_0, gamma_0, iter_0, acc0, accuracy_score(y_train, y_pred_trn_0))
	print(C_1, gamma_1, iter_1, acc1, accuracy_score(y_train, y_pred_trn_1))
	print(C_2, gamma_2, iter_2, acc2, accuracy_score(y_train, y_pred_trn_2))
	return [acc0,acc1, acc2]
		


def check_comp_complexity(Gs, ys, h, full_batch):
	begin = time.time()
	#print("number of graphs:", len(Gs))
	#Dw_s = LearnParametricDistance(Gs, ys, h, is_weight=False, train_index=train_index, test_index=test_index)
	learnOneEpoch(Gs, ys, h, full_batch = full_batch)
	end = time.time()
	return end - begin

def draw1(x, y, z):
	x = [2*e for e in x]
	plt.plot(x, y, color="red", marker="o", label="Stochastic")
	plt.plot(x,z, color="blue", marker="+", label="Full batch")
	plt.title("Running time of one epoch vs. Number of graphs")
	plt.xlabel("Number of graphs", fontsize=14)
	plt.ylabel("Running time in seconds", fontsize=14)
	#plt.grid=True
	plt.legend()
	plt.show()
def draw2(x,y,z):
	#x = [2*e for e in x]
	plt.plot(x, y, color="red", marker="o", label="Stochastic")
	plt.plot(x,z, color="blue", marker="+", label="Full batch")
	plt.title("Running time of one epoch vs. Number of WL iterations")
	plt.xlabel("Number of WL iterations", fontsize=14)
	plt.ylabel("Running time in seconds", fontsize=14)
	#plt.grid=True
	plt.legend()
	plt.show()

def run1():
	n_graphs_list=[10, 20, 50, 100, 200, 300, 400, 500]
	stochastic_times =[]
	full_batch_times=[]
	for ngraphs in n_graphs_list:
		Gs, labels =  simulationdata(ngraphs, 20, 0.3)
		ys = preprocess(labels)
		stime = check_comp_complexity(Gs, ys, h=2, full_batch=False)
		ftime = check_comp_complexity(Gs, ys, h=2, full_batch=True)
		
		stochastic_times.append(stime)
		full_batch_times.append(ftime)

		print(ngraphs, stime, ftime)
	draw1(n_graphs_list, stochastic_times,full_batch_times)

def run2():
	WL_n_iterations_list=[1,2,3,4,5,6,7]
	stochastic_times =[]
	full_batch_times=[]
	for wliterations in WL_n_iterations_list:
		Gs, labels =  simulationdata(50, 20, 0.3)
		ys = preprocess(labels)
		stime = check_comp_complexity(Gs, ys, h=wliterations, full_batch=False)
		ftime = check_comp_complexity(Gs, ys, h=wliterations, full_batch=True)
		
		stochastic_times.append(stime)
		full_batch_times.append(ftime)

		print(wliterations, stime, ftime)
	draw2(WL_n_iterations_list, stochastic_times,full_batch_times)

def main():
	#print("hello world")
	num_iterations = 3
	names=["MUTAG", "PTC_MR", "PROTEINS", "NCI1", "ENZYMES"]
	name = names[2]
	Gs, labels = load_data(name)

	#Gs, labels = simulationdata(50, 20,0.2)
	


	#run2()

	n = len(Gs)
	print("Dataset name: ", name)
	print("Number of graph-label pairs: ", len(Gs))

	# convert labels to {-1,1}
	print(labels)
	ys = np.array(labels)
	#ys = preprocess(labels)


	# computing WL subtree kernels with different H {1,2,...,n_iters}
	WLSubtreeKernels = computeWLSubTreeKernels(Gs, num_iterations)
	#randomwalkKernel = computeRandomWalkKernel(Gs)
	#print("RW done...")
	#shortestpathKernel = shortest_path_kernel_matrix(Gs)
	#print("SP done...")

	#randomwalkKernels=[]
	#shortestpathKernels=[]
	#for i in range(num_iterations):
	#	randomwalkKernels.append(randomwalkKernel)
	#	shortestpathKernels.append(shortestpathKernel)

	# Running training
	begin = time.time()
	run_cv_for_training(Gs, WLSubtreeKernels,ys, num_iterations)
	end = time.time()
	print("total running time of the program is :", end - begin)




if __name__ == '__main__':
    main()