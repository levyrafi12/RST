import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from features import extract_features
from relations_inventory import ind_to_action_map

import sklearn

def train_model(model_name, trees, samples, vocab, tag_to_ind_map):
	if model_name == "neural":
		model = neural_network_model(trees, samples, vocab, tag_to_ind_map)
	else: 
		model = linear_model(trees, samples, vocab, tag_to_ind_map)
	
	return model
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
	iterations=400, subset_size=5000, print_every=10):

	num_classes = len(ind_to_action_map)

	[x_vecs, _] = extract_features(trees, samples, vocab, 1, tag_to_ind_map)

	print("num features {}, num classes {}, num samples {}".\
		format(len(x_vecs[0]), num_classes, len(samples)))
	print("Running neural model")

	net = Network(len(x_vecs[0]), hidden_size, num_classes)
	print(net)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=lr)
	print(optimizer)

	tot = 0
	n_match = 0
	for i in range(1, iterations + 1):
		[x_vecs, y_labels] = extract_features(trees, samples, vocab, \
			min(subset_size, len(samples)), tag_to_ind_map)

		optimizer.zero_grad() # zero the gradient buffers
		# A two dimension array of size num samples * num of actions
		scores = net(Variable(torch.tensor(x_vecs, dtype=torch.float)))
		# indices[i] is the predicted action (ind) of sample i 
		indices = scores.max(1)[1] # dim=1 relates to rows, size = subset_size (batch size)
		loss = criterion(scores, Variable(torch.tensor(y_labels, dtype=torch.long)))
		loss.backward()
		optimizer.step()

		n_match += np.sum([indices[i] == y_labels[i] for i in range(len(indices))])
		tot += len(indices)
		if i % print_every == 0:
			print("t {0} num matches = {1:.3f}% loss {2:.3f}".\
				format(i, n_match / tot * 100, loss.item()))
			tot = 0
			n_match = 0

	# for param in net.parameters():
	# print(param.data)

	return net

def neural_net_predict(net, x_vecs):
	scores = net(Variable(torch.tensor(x_vecs, dtype=torch.float)))
	scores = log_softmax(scores)
	[sorted_scores, indices] = scores.sort(descending=True)
	sorted_actions = [ind_to_action_map[indices[i]] for i in range(len(indices))]
	return [scores, sorted_scores, sorted_actions]

def log_softmax(x):
	numerator = (x - x.max()).exp()
	return torch.log(numerator / numerator.sum())

def linear_model(trees, samples, vocab, tag_to_ind_map, \
	iterations=400, subset_size=500, print_every=50):

	[x_vecs, _] = extract_features(trees, samples, vocab, 1, tag_to_ind_map, word_encoding('linear'))

	y_all = list(range(len(ind_to_action_map)))

	print("num features {}, num classes {}, num samples {}".\
		format(len(x_vecs[0]), len(y_all), len(samples)))

	print("Running linear model")

	# tol=1e-7, learning_rate='constant', eta0=0.1
	clf = sklearn.linear_model.SGDClassifier()
	print(clf)

	tot = 0
	n_match = 0
	for i in range(1, iterations + 1):
		[x_vecs, y_labels] = extract_features(trees, samples, vocab, \
			min(subset_size, len(samples)), tag_to_ind_map, word_encoding('linear'))

		clf.partial_fit(x_vecs, y_labels, y_all)
		y_pred = clf.predict(x_vecs)
		n_match += np.sum([y_pred[i] == y_labels[i] for i in range(len(y_labels))])
		tot += len(y_labels)
		if i % print_every == 0:
			print("mini batch iter {0} num matches {1:.3f}%".format(i, n_match / tot * 100))
			tot = 0
			n_match = 0

	return clf

def linear_predict(clf, x_vecs):
	scores = clf.decision_function(x_vecs)[0]
	sorted_scores = np.sort(scores)[::-1]
	indices = clf.classes_[np.argsort(scores)][::-1]
	sorted_actions = [ind_to_action_map[indices[i]] for i in range(len(indices))]
	return scores, sorted_scores, sorted_actions


def word_encoding(model_name):
	if model_name == 'neural':
		return 'embedd'
	return 'one_hot'
