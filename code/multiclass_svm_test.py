import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

from multiclass_svm import MulticlassSVM

X = np.array([[0,0,0,1], [1,0,0,0], [0,1,0,0], [0,0,1,0],[0,0,1,1]])
y = np.array([0, 0, 1, 1,1])

test_x = np.array([[0,0,0,1]])

clf = MulticlassSVM(C=1, tol=0.001, max_iter=500, random_state=0, verbose=0)
clf.fit(X, y)

print(clf._label_encoder.classes_)
print(clf._label_encoder.transform([6]))
print("predict {}".format(clf.predict(test_x)))
print(clf.coef_)
print(clf.dual_coef_)