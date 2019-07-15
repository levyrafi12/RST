from sklearn import linear_model
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from features import extract_features
from relations_inventory import ind_to_action_map

hidden_size = 128
lr = 1e-4 # learning rate
momentum = 0.9

class Network(nn.Module):
    def __init__(self, n_features, hidden_size, num_classes):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden_size)
        # self.fc1.weight.data.fill_(1.0)
        nn.init.normal_(self.fc1.weight.data)
        self.fc1.weight.data *= np.sqrt(2 / n_features)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # self.fc2.weight.data.fill_(1.0)
        nn.init.normal_(self.fc2.weight.data)
        self.fc2.weight.data *= np.sqrt(2 / hidden_size)
        self.fc2.bias.data.fill_(0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))
 
def neural_network_model(trees, samples, vocab, tag_to_ind_map, \
	iterations=100, subset_size=5000):

	num_classes = len(ind_to_action_map)

	[x_vecs, _] = extract_features(trees, samples, vocab, \
		1, tag_to_ind_map)

	print("num features {}, num classes {}, num samples {}".\
		format(len(x_vecs[0]), num_classes, len(samples)))
	print("Running neural model")

	net = Network(len(x_vecs[0]), hidden_size, num_classes)
	print(net)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=lr)
	print(optimizer)

	for i in range(iterations):
		[x_vecs, y_labels] = extract_features(trees, samples, vocab, \
			min(subset_size, len(samples)), tag_to_ind_map)

		optimizer.zero_grad() # zero the gradient buffers
		# A two dimension array of size num samples * num of actions
		scores = net(Variable(torch.tensor(x_vecs, dtype=torch.float)))
		# indices[i] is the predicted action (ind) of sample i 
		indices = scores.max(1)[1] # dim=1 relates to rows, size = subset_size (batch size)
		n_match = np.sum([indices[i] == y_labels[i] for i in range(len(indices))])
		print("num matches = {}%".format(n_match / len(indices) * 100))
		loss = criterion(scores, Variable(torch.tensor(y_labels, dtype=torch.long)))
		print("t = {} loss = {}".format(i, loss.item()))
		loss.backward()
		optimizer.step()

	print("t = {} loss = {}".format(iterations, loss.item()))
	# for param in net.parameters():
	# print(param.data)

	return net

def neural_net_predict(net, x_vecs):
	scores = net(Variable(torch.tensor(x_vecs, dtype=torch.float)))
	scores = softmax(scores)
	[sorted_scores, indices] = scores.sort(descending=True)
	sorted_actions = [ind_to_action_map[indices[i]] for i in range(len(indices))]
	return [scores, sorted_scores, sorted_actions]

def softmax(x):
	numerator = (x - x.max()).exp()
	return numerator / numerator.sum()

def mini_batch_linear_model(trees, samples, y_all, vocab, \
	max_edus, tag_to_ind_map, iterations=200, subset_size=500):

	print("n_samples = {}, n_classes = {}".format(len(samples), len(y_all)))
	print("Running linear model")

	classes = y_all

	clf = linear_model.SGDClassifier(tol=1e-7, learning_rate='constant', eta0=0.1)
	print(clf)

	for i in range(iterations):
		[x_vecs, y_labels] = extract_features(trees, samples, vocab, \
			subset_size, max_edus, tag_to_ind_map)

		dec = linear_train(clf, x_vecs, y_labels, classes)
		# scores = [y_all[np.argmax(elem)] for elem in dec]
		# n_match = np.sum([scores[i] == y_labels[i] for i in range(len(scores))])
		# print("num matches = {}%".format(n_match / len(scores) * 100))
		classes = None

	return clf

def linear_train(clf, x_vecs, y_labels, classes):
	clf.partial_fit(x_vecs, y_labels, classes)
	dec = clf.decision_function(x_vecs)
	return dec

def linear_predict(clf, x_vecs):
	return clf.predict(x_vecs)
