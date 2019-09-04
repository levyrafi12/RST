import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import sys
sys.stdout.flush()

from features import extract_features
from features import extract_features_next_subset
from features import get_word_encoding
from features import is_bag_of_words
from features import is_basic_feat
from features import project_features
from relations_inventory import ind_to_action_map
from rst_parser import evaluate
from dplp import dplp_algo
from model_defs import Model
from sequence import sequence_model

import sklearn
import math

def train_model(model_name, trees, samples, sents, pos_tags, vocab, tag_to_ind_map, \
	gen_dep, max_sent_len):
	model = Model(model_name)
	if model_name == "neural":
		neural_network_model(model, trees, samples, vocab, tag_to_ind_map, gen_dep)
	elif model_name == "dplp":
		dplp_model(model, trees, samples, vocab, tag_to_ind_map)
	elif model_name == "seq":
		sequence_model(model, trees, samples, sents, pos_tags, vocab, tag_to_ind_map, \
			max_sent_len)
	else:
		linear_model(model, trees, samples, vocab, tag_to_ind_map, gen_dep)
	return model

hidden_size = 128
lr = 1e-4 # learning rate
momentum = 0.9

class Network(nn.Module):
    def __init__(self, n_features, hidden_size, num_classes):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden_size)
        nn.init.normal_(self.fc1.weight.data)
        self.fc1.weight.data *= np.sqrt(2 / n_features)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.fc2.weight.data)
        self.fc2.weight.data *= np.sqrt(2 / hidden_size)
        self.fc2.bias.data.fill_(0.0)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        nn.init.normal_(self.fc3.weight.data)
        self.fc3.weight.data *= np.sqrt(2 / hidden_size)
        self.fc3.bias.data.fill_(0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
 
def neural_network_model(model, trees, samples, vocab, tag_to_ind_map, \
	gen_dep, n_epoch=20, subset_size=64, print_every=1):
	num_classes = len(ind_to_action_map)
	subset_size = min(subset_size, len(samples))

	[x_vecs, _] = extract_features(trees, samples, vocab, 1, tag_to_ind_map)

	print("num features {}, num classes {}, num samples {} subset size {} vocab {}".\
		format(len(x_vecs[0]), num_classes, len(samples), subset_size), len(vocab))
	print("Running neural model")

	net = Network(len(x_vecs[0]), hidden_size, num_classes)
	model._clf = net
	print(net)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=lr)
	print(optimizer)

	n_match = 0

	# grad_updates = n_epoch * (n_samples / subset_size)

	for epoch in range(1, n_epoch + 1):
		for [x_vecs, y_labels] in extract_features_next_subset(\
			trees, samples, vocab, subset_size, tag_to_ind_map):
			optimizer.zero_grad() # zero the gradient buffers
			# A two dimension array of size num samples * num of actions
			scores = net(Variable(torch.tensor(x_vecs, dtype=torch.float)))
			# indices[i] is the predicted action (ind) of sample i 
			indices = scores.max(1)[1] # dim=1 refers to rows, size = subset_size (batch size)
			loss = criterion(scores, Variable(torch.tensor(y_labels, dtype=torch.long)))
			loss.backward()
			optimizer.step()

			n_match += np.sum([indices[j] == y_labels[j] for j in range(len(indices))])
		if epoch % print_every == 0:
			print("epoch {0} num matches = {1:.3f}% loss {2:.3f}".\
				format(epoch, n_match / len(samples) * 100, loss.item()))
			n_match = 0
		evaluate(model, vocab, tag_to_ind_map, gen_dep)
		gen_dep = False

	# for param in net.parameters():
	# print(param.data)

def dplp_model(model, trees, samples, vocab, tag_to_ind_map):
	dplp_algo(model, trees, samples, vocab, tag_to_ind_map)
	# linear_model(model, trees, samples, vocab, tag_to_ind_map)

def linear_model(model, trees, samples, vocab, tag_to_ind_map, \
	gen_dep, n_epoch=10, subset_size=64, print_every=1):

	[x_vecs, _] = extract_features(trees, samples, vocab, 1, tag_to_ind_map, \
		is_bag_of_words(model._name), is_basic_feat(model._name), \
		get_word_encoding(model._name))

	if model.is_proj_mat():
		x_vecs = project_features(model._proj_mat, x_vecs)

	y_all = list(range(len(ind_to_action_map)))
	subset_size = min(subset_size, len(samples))

	print("num features {}, num classes {}, num samples {} subset size {} vocab {}".\
		format(len(x_vecs[0]), len(y_all), len(samples), subset_size, len(vocab)))

	print("Running {} model 'word encoding' {} 'bag of words' {} 'basic feat' {}".\
		format(model._name, get_word_encoding(model._name), \
		is_bag_of_words(model._name), is_basic_feat(model._name)))

	clf = sklearn.linear_model.SGDClassifier()
	model._clf = clf
	print(clf)

	n_match = 0
	n_subsets = math.ceil(len(samples) / subset_size)
	n_samples_in_epoch = n_subsets * subset_size

	for epoch in range(1, n_epoch + 1):
		for i in range(n_subsets):
			[x_vecs, y_labels] = extract_features(trees, samples, vocab, \
				subset_size, tag_to_ind_map, is_bag_of_words(model._name),\
				is_basic_feat(model._name), get_word_encoding(model._name))

			if model.is_proj_mat():
				x_vecs = project_features(model._proj_mat, x_vecs)
	
			clf.partial_fit(x_vecs, y_labels, y_all)
			y_pred = clf.predict(x_vecs)
			n_match += np.sum([y_pred[j] == y_labels[j] for j in range(len(y_labels))])
		if epoch % print_every == 0:
			print("epoch {0} num matches {1:.3f}%".format(\
				epoch, n_match / n_samples_in_epoch * 100))
			n_match = 0
		evaluate(model, vocab, tag_to_ind_map, gen_dep)
		gen_dep = False
