import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from numpy import linalg as LA

from sklearn import svm

from features import extract_features
from features import is_basic_feat
from features import get_word_encoding
from relations_inventory import ind_to_action_map
from features import project_features
from model_defs import Model

# from sklearn import svm
from multiclass_svm import MulticlassSVM

def dplp_algo(model, trees, samples, vocab, tag_to_ind_map, subset_size=500, print_every=10):
	C = 1.0 # {1, 10, 50, 100}
	# as tau becomes smaller the effect of A_prev is larger. No effect when tau = 1
	tau = 1.0 # tau = { 1, 0.1, 0.01, 0.001} 
	K = 30 # { 30, 60, 90, 150}

	subset_size = min(subset_size, len(samples))

	[x_vecs, _] = extract_features(trees, samples, vocab, 1, tag_to_ind_map, \
		True, is_basic_feat(model._name), get_word_encoding(model._name))

	n_features = len(x_vecs[0])
	
	print("vocab size {}, num features {}, num classes {}, num samples {} subset size {}".\
		format(len(vocab._tokens), n_features, len(ind_to_action_map), len(samples), subset_size))

	print("Running dplp model")
	A_t_1 =  np.random.uniform(0, 1, (K, len(x_vecs[0]))) # A(t - 1)

	T = 200
	eps = 0.001 
	clf = MulticlassSVM(C=C, tol=0.01, max_iter=100, random_state=0, verbose=0)

	for t in range(1, T + 1):
		if t > 0 and t % print_every == 0:
			print("t {}".format(t))
		[x_vecs, y_labels] = extract_features(trees, samples, vocab, subset_size, \
			tag_to_ind_map, True, is_basic_feat(model._name), \
			get_word_encoding(model._name))
		Av = project_features(A_t_1, x_vecs) # Av dim is subset_size * K
		clf.fit(Av, y_labels)
		A_t = solve_proj_mat_iter(clf, A_t_1, t, tau, x_vecs, y_labels)
		if t == 2:
			A2_diff = LA.norm(A_t - A_t_1)
		elif t > 2:
			A_diff = LA.norm(A_t - A_t_1)
			if t % print_every == 0:
				print("Ratio {0:.4f}".format(A_diff / A2_diff))
			if A_diff / A2_diff < eps:
				break
		A_t_1 = A_t

	[x_vecs, y_labels] = extract_features(trees, samples, vocab, \
		subset_size, tag_to_ind_map, True,\
		is_basic_feat(model._name), get_word_encoding(model._name))
	x_vecs = project_features(A_t, x_vecs)

	clf.fit(x_vecs, y_labels)
	# clf = svm.SVC(C=C, kernel='linear', decision_function_shape='ovr')
	clf.fit(x_vecs, y_labels)
	model._proj_mat = A_t
	model._clf = clf

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
		[y_i] = le.transform([y_labels[i]])
		expected_weight = sum([(delta_f(m, y_i) - alpha[m, i]) * w[m, :] for m in range(n_classes)])
		A += np.outer(w[y_i, :] - expected_weight, x_vecs[i])

	A *= (1 / t)
	A += (1 - tau / t) * A_prev
	return A

def delta_f(m, y_i):
	if y_i == m:
		return 1
	return 0