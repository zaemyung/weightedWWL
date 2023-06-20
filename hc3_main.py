import argparse
import pickle
import os
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import manifold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold
from sklearn.model_selection._validation import _fit_and_score
from sklearn.svm import SVC

from generaterandomgraphs import load_hc3_dataset, simulationdata
from kernel import *
from utils import *

def grid_search_cv(model, param_grid, precomputed_kernels, y, cv=5):
	cv = StratifiedKFold(n_splits=cv, shuffle=False)
	results =[]
	for train_index, test_index in cv.split(precomputed_kernels[0], y):
		split_results = []
		params = []
		for idx, K in enumerate(precomputed_kernels):
			for p in list(ParameterGrid(param_grid)):
				sc_result = _fit_and_score(clone(model), K, y, scorer=make_scorer(accuracy_score), train=train_index,
                                           test=test_index, verbose=0, parameters=p,fit_params=None)
				sc = sc_result['test_scores']
				split_results.append(sc)
				params.append({'K_idx':idx, 'params':p})
		results.append(split_results)
	results=np.array(results)
	fin_results = results.mean(axis=0)
	best_idx = np.argmax(fin_results)
	print(best_idx, fin_results[best_idx])
	ret_model = clone(model).set_params(**params[best_idx]['params'])
	return ret_model.fit(precomputed_kernels[params[best_idx]['K_idx']], y), params[best_idx]


def run_cv_for_training(Gs, K0_s, ys, num_iterations):
	print("Running 5-cv...")
	cv = KFold(n_splits=5, shuffle=True)
	list_acc=[]
	index = 0
	for train_index, test_index in cv.split(K0_s[0], ys):
		index+=1
		D_s = LearnParametricDistance(Gs, ys, num_iterations, is_weight=False, train_index=train_index, test_index=test_index)
		Dw_s = LearnParametricDistance(Gs, ys, num_iterations, is_weight=True, train_index=train_index, test_index=test_index)
		#Dw_s = pairwise_distance(Gs, y, sinkhorn_lambda=0.01, kl_lambda=0.1, num_iterations = num_iterations, is_weight=True, train_index = train_index, test_index = test_index)
		##Dw_s = K0_s
		# print(f'len(D_s): {len(D_s)}\nlen(Dw_s):{len(Dw_s)}')
		# print('D_s[0].shape:', D_s[0].shape)
		# print(f'Dw_s[0]:{Dw_s[0]}')
		# print(f'{np.sum(Dw_s, axis=1)}')
		# print('---')
		# print(f'{np.sum(Dw_s[0], axis=0)}')
		# print(f'{np.sum(Dw_s[0], axis=1)}')
		# print(f'{np.sum(np.sum(Dw_s[0], axis=0), axis=0)}')
		# print(f'{np.sum(np.sum(Dw_s[0], axis=0), axis=0)}')

		accs = run_train_test(K0_s, D_s, Dw_s, ys, train_index, test_index, num_iterations)
		list_acc.append(accs)
		print("term ", index, "  SVM ", accs)
		#return 0
	print('np.mean of list_acc:', np.mean(list_acc, axis=0))


def compute_subtree_scores(Gs, K0_s, ys, num_iterations, domain):
    train_indices = list(range(len(K0_s[0])))
    test_indices = list(range(len(ys)))

    pkl_path = f'{domain}-{num_iterations}.pkl'
    if os.path.isfile(pkl_path):
        with open(pkl_path, 'rb') as f:
            Dw_s, ws, id2labels, wl = pickle.load(f)
    else:
        Dw_s, ws, id2labels, wl = LearnParametricDistance(Gs, ys, num_iterations, is_weight=True, train_index=train_indices, test_index=test_indices, return_weights=True)
        with open(pkl_path, 'wb') as f:
            pickle.dump((Dw_s, ws, id2labels, wl), f)

    assert len(ws) == len(id2labels)
    for i in range(len(ws)):
        print(f'h={i}')
        n = 3
        pos_sorted_inds = np.argsort(ws[i])[::-1][:n]
        neg_sorted_inds = np.argsort(ws[i])[:n]
        for pos_ind, neg_ind in zip(pos_sorted_inds, neg_sorted_inds):
            print(f'pos: {id2labels[i][pos_ind]}:{ws[i][pos_ind]}')
            print(f'neg: {id2labels[i][neg_ind]}:{ws[i][neg_ind]}')
    print(list(wl._label_dict.items()))


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
	# print('len(kernal_train_0):', len(K_train_0))
	# print('kernal_train_0[best K_idx]]:', K_train_0[best_params_0['K_idx']])
	# print('kernal_train_0[best K_idx]].shape:', K_train_0[best_params_0['K_idx']].shape)

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



def main():
    domains = ['finance', 'medicine', 'open_qa', 'wiki_csai', 'reddit_eli5']
    # domains = ['finance', 'medicine', 'open_qa', 'reddit_eli5']
    # domains = ['wiki_csai']
    for domain in domains:
        print('-----------------')
        print(domain)
        num_iterations = 3
        name = f'HC3-{domain}'
        Gs, labels = load_hc3_dataset(domain=domain)
        ys = np.array(labels)
        print('Dataset name: ', name)
        print('Number of graph-label pairs: ', len(Gs))

        WLSubtreeKernels = computeWLSubTreeKernels(Gs, num_iterations)
        print('WLSubtreeKernels[0].shape:', WLSubtreeKernels[0].shape)
        print('len(WLSubtreeKernels):', len(WLSubtreeKernels))

        # Running training
        begin = time.time()
        compute_subtree_scores(Gs, WLSubtreeKernels, ys, num_iterations, domain)
        # run_cv_for_training(Gs, WLSubtreeKernels, ys, num_iterations)

        end = time.time()
        print("total running time of the program is :", end - begin)


if __name__ == '__main__':
    main()