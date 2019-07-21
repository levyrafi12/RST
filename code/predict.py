import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from features import extract_features
from relations_inventory import ind_to_action_map

import sklearn
import math

def predict(model, model_name, x_vecs):
	if model_name == 'neural':
		return neural_net_predict(model, x_vecs) 
	return linear_predict(model, x_vecs)

def neural_net_predict(net, x_vecs):
	scores = net(Variable(torch.tensor(x_vecs, dtype=torch.float)))
	scores = log_softmax(scores)
	[sorted_scores, indices] = scores.sort(descending=True)
	sorted_actions = [ind_to_action_map[indices[i]] for i in range(len(indices))]
	return [scores, sorted_scores, sorted_actions]

def log_softmax(x):
	numerator = (x - x.max()).exp()
	return torch.log(numerator / numerator.sum())

def linear_predict(clf, x_vecs):
	scores = clf.decision_function([x_vecs])[0]
	sorted_scores = np.sort(scores)[::-1]
	indices = clf.classes_[np.argsort(scores)][::-1]
	sorted_actions = [ind_to_action_map[indices[i]] for i in range(len(indices))]
	return scores, sorted_scores, sorted_actions