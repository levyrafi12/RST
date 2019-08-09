import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from numpy import linalg as LA

from features import extract_features
from relations_inventory import ind_to_action_map

# from sklearn import svm
from multiclass_svm import MulticlassSVM

def dplp_algo(model_name, trees, samples, vocab, tag_to_ind_map, subset_size=500):
	C = 0.1
	tau = 1.0
	K = 30

	subset_size = min(subset_size, len(samples))

	[x_vecs, _] = extract_features(trees, samples, vocab, 1, tag_to_ind_map, \
		True, False)

	n_features = len(x_vecs[0])
	
	print("vocab size {}, num features {}, num classes {}, num samples {} subset size {}".\
		format(len(vocab._tokens), n_features, len(ind_to_action_map), len(samples), subset_size))

	print("Running dplp model")
	A_t_1 =  np.random.uniform(0, 1, (K, len(x_vecs[0])))

	T = 500
	eps = 0.01 
	# clf = svm.SVC(C=C, kernel='linear')
	clf = MulticlassSVM(C=C, tol=0.01, max_iter=100, random_state=0, verbose=0)

	for t in range(1, T + 1):
		if t % 10:
			print("t {}".format(t))
		[x_vecs, y_labels] = extract_features(trees, samples, vocab, subset_size, \
			tag_to_ind_map, True, False)
		v = np.array(x_vecs).T
		Av = np.matmul(A_t_1, v)
		clf.fit(Av.T, y_labels)
		A_t = solve_proj_mat_iter(clf, A_t_1, t, tau, x_vecs, y_labels)
		if t == 2:
			A2_diff = LA.norm(A_t - A_t_1)
		elif t > 2:
			A_diff = LA.norm(A_t - A_t_1)
			if t % 10:
				print("Ratio {0:.4f}".format(A_diff / A2_diff))
			if A_diff / A2_diff < eps:
				break
		A_t_1 = A_t

	[x_vecs, y_labels] = extract_features(trees, samples, vocab, len(samples), \
		tag_to_ind_map, True, False)
	v = np.array(x_vecs).T
	Av = np.zeros((A_t.shape[0], v.shape[1]))
	for i in range(A_t.shape[0]):
		for j in range(v.shape[1]):
			Av[i,j] = np.matmul(A_t[i, :], v[:, j])
	clf.fit(Av.T, y_labels)

	return A_t, clf

def solve_proj_mat_iter(clf, A_prev, t, tau, x_vecs, y_labels):
	"""
		Solve projection matrix A iteratively
	"""
	le = clf._label_encoder
	n_classes = len(le.classes_)
	n_vec = len(y_labels)

	alpha = clf.dual_coef_
	w = clf.coef_
	A = np.zeros(A_prev.shape)

	for i in range(n_vec):
		expected_weight = sum([alpha[m, i] * w[m, :] for m in range(n_classes)])
		[y_i] = le.transform([y_labels[i]])
		A += np.outer(w[y_i, :] - expected_weight, x_vecs[i])

	A *= (1 / t)
	A += (1 - tau / t) * A_prev
	return A

