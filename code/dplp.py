import numpy as np
from numpy import linalg as LA

from features import extract_features
from relations_inventory import ind_to_action_map
from rst_parser import evaluate

from sklearn import svm

def dplp_algo(model_name, trees, samples, vocab, tag_to_ind_map, subset_size=500):
	C = 1.0
	tau = 1.0
	K = 60

	num_classes = len(ind_to_action_map)
	subset_size = min(subset_size, len(samples))

	[x_vecs, _] = extract_features(trees, samples, vocab, 1, tag_to_ind_map, \
		None, True, True)

	print("num features {}, num classes {}, num samples {} subset size {}".\
		format(len(x_vecs[0]), num_classes, len(samples), subset_size))

	print("Running dplp model")
	A_t_1 =  np.random.uniform(0,1, (K, len(x_vecs[0])))

	T = 10 
	eps = 0.1 
	clf = svm.SVC(C=C, kernel='linear')
	print(clf)
	A2_diff = 0

	for t in range(1, T + 1):
		[x_vecs, y_labels] = extract_features(trees, samples, vocab, subset_size, tag_to_ind_map, \
			None, True, True)
		v = np.array(x_vecs).T
		Av = np.matmul(A_t_1, v)
		clf.fit(Av.T, y_labels)
		A_t = solve_proj_mat_iter(clf, A_t_1, t, tau, x_vecs, y_labels)
		if t == 2:
			A2_diff = LA.norm(A_t - A_t_1)
		elif t > 2:
			A_diff = LA.norm(A_t - A_t_1)
			if A_diff / A2_diff < Eps:
				break
		A_t_1 = A_t

	[x_vecs, y_labels] = extract_features(trees, samples, vocab, len(samples), tag_to_ind_map, \
		None, True, True)
	Av = np.multiply(A_t, x_vecs)
	clf.fit(Av, y_labels)

	return A_t, clf

def solve_proj_mat_iter(clf, A_prev, t, tau, x_vecs, y_labels):
	"""
		Solve projection matrix A iteratively
	"""
	print(np.unique(y_labels))
	num_classes = len(np.unique(y_labels))

	eta = clf.dual_coef_
	w = clf.coef_
	sv = clf.support_vectors_
	sv_ind = clf.support_
	n_sv = clf.n_support_
	A = np.zeros(A_prev.shape)

	print("len y labels unique {}".format(len(np.unique(y_labels))))
	print("dual vars {}".format(eta.shape))
	print("weights {}".format(w.shape))
	print("support vectors {}".format(sv.shape))
	print("sv indices {}".format(len(sv_ind)))
	print("n sv {}".format(n_sv))

	for i in sv_ind:
		expected_weight = sum([eta[i,m] * w[m,:] for m in num_classes])
		print("expected weight {}".format(expected_weight))
		true_lable_weight = w[:,y_labels[i]]
		print(true_lable_weight)
		weight_diff = true_lable_weight - expected_weight
		print(weight_diff)
		A += weight_diff * x_vecs[i]

	alpha_t = 1 / t
	A *= alpha_t
	A += (1 - alpha_t * tau) * A_prev
	return A