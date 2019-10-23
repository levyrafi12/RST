import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from features import extract_features
from relations_inventory import ind_to_action_map
from relations_inventory import baseline_action

from multiclass_svm import MulticlassSVM

from model_defs import Model
from features import project_features

import sklearn
import math

def predict(model, x_vecs):
	with torch.no_grad():
		if model._name == 'neural' or model._name == 'seq':
			return neural_net_predict(model, x_vecs) 
		if model._name == 'dplp':
			return dplp_predict(model, x_vecs)
		return linear_predict(model, x_vecs)
	
def neural_net_predict(model, x_vecs):
	net = model._clf
	scores = net(Variable(torch.tensor(x_vecs, dtype=torch.float)))
	scores = log_softmax(scores)
	[sorted_scores, indices] = scores.sort(descending=True)
	sorted_actions = [ind_to_action_map[indices[i]] for i in range(len(indices))]
	return [scores, sorted_scores, sorted_actions]

def log_softmax(x):
	numerator = (x - x.max()).exp()
	return torch.log(numerator / numerator.sum())

def linear_predict(model, x_vecs):
	clf = model._clf
	if model.is_proj_mat():
		[x_vecs] = project_features(model._proj_mat, [x_vecs]).tolist()
	scores = clf.decision_function([x_vecs])[0]
	sorted_scores = np.sort(scores)[::-1]
	indices = clf.classes_[np.argsort(scores)][::-1]
	sorted_actions = [ind_to_action_map[indices[i]] for i in range(len(indices))]
	return scores, sorted_scores, sorted_actions

def dplp_predict(model, x_vecs):
	A = model._proj_mat
	clf = model._clf
	x_vecs = project_features(A, [x_vecs])
	[predict] = clf.predict(x_vecs)
	actions = [ind_to_action_map[predict]]
	scores = [0] * len(ind_to_action_map)
	# Correcting illegal action 
	if actions[0] == 'SHIFT':
		actions.append(baseline_action)

	return scores, scores, actions
