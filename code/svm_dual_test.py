import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn import svm
from sklearn import multiclass
import numpy as np

def class_w_max_votes(dec, n_classes):
	votes = np.zeros(n_classes)

	k = 0
	for i in range(n_classes):
		for j in range(i + 1, n_classes):
			if dec[k] > 0:
				votes[i] += 1
			else:
				votes[j] += 1
		k += 1

	return np.argmax(votes)

X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]

test_x = [[4]]
n_classes = len(np.unique(Y))

clf = svm.SVC(kernel='linear', decision_function_shape='ovo', probability=True)
"""
clf1 = multiclass.OneVsRestClassifier(clf)
print(clf1)

clf1.fit(X, Y)
print(clf1.coef_)
print(clf1.intercept_)
dec = clf1.decision_function([[0]])
print(dec)
"""

clf.fit(X, Y)
"""
print(clf.coef_)
print(clf.intercept_)
print(clf.dual_coef_)
print(clf.support_vectors_)
print(clf.support_)
print(clf.n_support_)
"""

dec = clf.decision_function(test_x)
pred = clf.predict(test_x)
print("dec {}".format(dec))
print("pred {}".format(pred))
print("class {}".format(class_w_max_votes(dec[0,:], n_classes)))

clf.decision_function_shape = "ovr"
dec = clf.decision_function(test_x)
pred = clf.predict(test_x)
print("dec {}".format(dec))
print("pred {}".format(pred))

print("sum dual coef {}".format(clf.dual_coef_.sum()))

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



