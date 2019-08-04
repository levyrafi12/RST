import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn import svm
from sklearn import multiclass
import numpy as np

X = [[0,1], [1,1], [2,1], [3,3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(kernel='linear', decision_function_shape='ovo', probability=True)
clf.fit(X, Y)
print(clf.coef_)
print(clf.intercept_)
print(clf.dual_coef_)
print(clf.support_vectors_)
print(clf.support_)
print(clf.n_support_)
dec = clf.decision_function([[3,3]])
print(dec)
clf.decision_function_shape = "ovr"
dec = clf.decision_function([[3,3]])
print(dec)

n_class = len(np.unique(Y))
n_features = len(X[0])
n_bin_class = int(n_class * (n_class - 1) / 2)

w = np.zeros((n_bin_class, n_features), dtype=float)
row = 0

x_feat = np.array(X)

support_indices = np.cumsum(clf.n_support_)

sv_ind1 = range(0, support_indices[0])

for i in range(n_class):
	if i > 0:
		sv_ind1 = range(support_indices[i - 1], support_indices[i])

	for j in range(i + 1, n_class):
		sv_ind2 = range(support_indices[j - 1], support_indices[j])

		for k in sv_ind1:
			alpha = clf.dual_coef_[j - 1, k]
			l = clf.support_[k]
			w[row, :] += alpha * x_feat[l, :]

		for k in sv_ind2:
			alpha = clf.dual_coef_[i, k]
			l = clf.support_[k]
			w[row, :] += alpha * x_feat[l, :]
		row += 1

assert clf.coef_.all() == w.all(), "matrices differ from each other"


