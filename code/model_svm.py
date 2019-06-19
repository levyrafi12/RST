from sklearn import svm
from sklearn import linear_model
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math

from features import extract_features
from relations_inventory import ind_to_action_map

svm_max_iter = 3000
svm_verbose = False

hidden_size = 128
lr = 1e-4 # learning rate

class Network(nn.Module):
    def __init__(self, n_features, hidden_size, num_classes):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden_size)
        self.fc1.weight.data.fill_(1.0)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.fc2.weight.data.fill_(1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))
 
def neural_network_model(trees, samples, vocab, max_edus, tag_to_ind_map, \
	iterations=2, subset_size=5000, print_every=10):

	num_classes = len(ind_to_action_map)

	[x_vecs, _] = extract_features(trees, samples, vocab, \
		1, max_edus, tag_to_ind_map)

	print("Running neural model")

	# x = torch.randn(subset_size, len(x_vecs[0]))
	# y = torch.randn(subset_size, num_classes)

	print("num features {}".format(len(x_vecs[0])))

	net = Network(len(x_vecs[0]), hidden_size, num_classes)
	print(net)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

	for i in range(iterations):
		[x_vecs, y_labels] = extract_features(trees, samples, vocab, \
			subset_size, max_edus, tag_to_ind_map)

		# print("{} {}".format(x_vecs, y_labels))

		# print("requires grad in fc1 and fc2 {} {}".format(net.fc1.weight.requires_grad,
		# net.fc2.weight.requires_grad))

		# print("data {} {}".format(net.fc1.weight.data, net.fc2.weight.data))

		# print("input requires grad {}".format(x.requires_grad))

		y_pred = net(Variable(torch.tensor(x_vecs, dtype=torch.float)))
		# y_pred = net(x)

		# print(y_pred)
		# y = set_one_hot(y_labels, num_classes)
		# print("y {}".format(y))

		# print("y_pred {} y {}".format(y_pred.requires_grad, y.requires_grad))

		# print("{} {}".format(y_pred.shape, y.shape))

		scores = y_pred.data.max(1)[1]
		# print(y_labels)

		# print("{} {}".format(len(scores), len(y_labels)))
		n_match = np.sum([scores[i] == y_labels[i] for i in range(len(scores))])
		# print("num matches = {}%".format(n_match / len(scores) * 100))

		loss = criterion(y_pred, Variable(torch.tensor(y_labels, dtype=torch.long)))
		# print("loss requires_grad {}".format(loss.data.requires_grad))

		# print("t = {} loss = {}".format(i, loss.item()))

		# if i > 0 and i % print_every == 0:
		# print("mini batch iter = {} , loss = {}".format(i, loss.item()))

		optimizer.zero_grad()   # zero the gradient buffers
		loss.backward()
		optimizer.step()

	return net

def neural_net_predict(net, x_vecs):
	return net(Variable(torch.tensor(x_vecs, dtype=torch.float)))

def mini_batch_linear_model(trees, samples, y_all, vocab, \
	max_edus, tag_to_ind_map, iterations=200, subset_size=500, print_every=10):

	print("n_samples = {} , vocab size = {} , n_classes = {}".\
		format(len(samples), len(vocab._tokens), len(y_all)))

	print("Running linear model")

	classes = y_all

	clf = linear_model.SGDClassifier(tol=1e-7, learning_rate='constant', eta0=0.1)
	print(clf)

	for i in range(iterations):
		if False and i > 0 and i % print_every == 0:
			print("mini batch iter = {}".format(i))

		[x_vecs, y_labels] = extract_features(trees, samples, vocab, \
			subset_size, max_edus, tag_to_ind_map)

		dec = linear_train(clf, x_vecs, y_labels, classes)
		scores = [y_all[np.argmax(elem)] for elem in dec]
		n_match = np.sum([scores[i] == y_labels[i] for i in range(len(scores))])
		# print("num matches = {}%".format(n_match / len(scores) * 100))
		classes = None

	return clf

def linear_train(clf, x_vecs, y_labels, classes):
	clf.partial_fit(x_vecs, y_labels, classes)
	dec = clf.decision_function(x_vecs)
	return dec

def linear_predict(clf, x_vecs):
	return clf.predict(x_vecs)


def non_linear_model(trees, samples, vocab, max_edus, tag_to_ind_map, \
	iterations=1000, subset_size=5000, print_every=10):

	print("n_samples = {} , vocab size = {}".format(len(samples), len(vocab)))

	# clf_lin = svm.LinearSVC(verbose=svm_verbose, max_iter=svm_max_iter)
	clf = svm.SVC(verbose=svm_verbose, max_iter=svm_max_iter, 
		decision_function_shape='ovr')

	print(clf)

	for i in range(iterations):
		if i > 0 and i % print_every == 0:
			print("mini batch iter = {}".format(i))

		[x_vecs, y_labels] = extract_features(trees, samples, vocab, \
			subset_size, max_edus, tag_to_ind_map)
		dec = non_linear_train(clf, x_vecs, y_labels)
		classes = np.unique(y_labels)
		scores = [classes[np.argmax(elem)] for elem in dec]
		n_match = np.sum([scores[i] == y_labels[i] for i in range(len(scores))])
		print("num matches = {}".format(n_match / len(scores) * 100))

def non_linear_train(clf, x_vecs, y_labels):
	# print("n_features = {}".format(len(x_vecs[0])))
	clf.fit(x_vecs, y_labels)
	if hasattr(clf, "n_support_"):
		arr = clf.n_support_
		ind = np.argmax(arr)
		print("{} {} {}".format(np.sum(arr), ind_to_action_map[y_labels[np.argmax(arr)]], np.max(arr)))
		print("count {}".format(y_labels.count(y_labels[ind])))

	dec = clf.decision_function(x_vecs)
	# print(dec.shape)
	return dec

def log_softmax(x):
	print(x.shape)
	numerator = np.exp(x - np.max(x))
	x = numerator / np.sum(numerator)
	# x = [math.log(x[i]) for i in range(len(x))] 
	return x

