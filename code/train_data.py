from preprocess import Node
from preprocess import TreeInfo
from utils import map_to_cluster
from glove import loadWordVectors
from relations_inventory import action_to_ind_map
from relations_inventory import ind_to_action_map
from vocabulary import gen_vocabulary
from preprocess import build_file_name
from preprocess import SEP

import glob
import copy

import numpy as np

import sys

class Sample(object):
	def __init__(self, tree):
		self._state = [] # [v1, v2, v3] where v1 & v2 are the elements at the top of the stack
		self._spans = [] # [[s1,t1],[s2,t2],[s3,t3]]
		self._encoded_spans = []
		self._action = ''
		self._tree = tree
		self._sents_spans = []

	def print_info(self):
		print("sample {} {}".format(self._state, self._action))

def gen_train_data(trees, model, path, print_data=False):
	samples = []
	sents = []
	pos_tags = []

	for tree in trees:
		fn = build_file_name(tree._fname, path, "TRAINING", "out.edus")
		root = tree._root
		stack = []
		tree_samples = []
		queue = [] # queue of EDUS indices
		for j in range(tree._root._span[1]):
			queue.append(j + 1)

		queue = queue[::-1]
		gen_train_data_tree(tree, root, stack, queue, tree_samples, model._stack_depth)
		tree._samples = tree_samples

		if print_data:
			outfn = path
			outfn += SEP + "train_data" + SEP
			outfn += tree._fname
			with open(outfn, "w") as ofh:
				for sample in tree_samples:
					ofh.write("{} {}\n".format(sample._state, sample._action))

	for tree in trees:
		for sample in tree._samples:
			samples.append(sample)
			
	y_all = [action_to_ind_map[samples[i]._action] for i in range(len(samples))]
	y_all = np.unique(y_all)

	return [samples, y_all]
					
def gen_train_data_tree(tree, node, stack, queue, samples, stack_depth):
	# node.print_info()
	sample = Sample(tree)
	if node._type == "leaf":
		sample._action = "SHIFT"
		sample._state, sample._spans, sample._sents_spans = \
			gen_state(tree, stack, queue, stack_depth)
		assert(queue.pop(-1) == node._span[0])
		stack.append(node)
	else:
		[l, r] = node._childs
		gen_train_data_tree(tree, l, stack, queue, samples, stack_depth)
		gen_train_data_tree(tree, r, stack, queue, samples, stack_depth)
		if r._nuclearity == "Satellite":
			sample._action = gen_action(node, r)
		else:
			sample._action = gen_action(node, l)
	
		sample._state, sample._spans, sample._sents_spans = \
			gen_state(tree, stack, queue, stack_depth)
		assert(stack.pop(-1) == node._childs[1])
		assert(stack.pop(-1) == node._childs[0])
		stack.append(node)

	samples.append(sample)

def gen_action(parent, child):
	action = "REDUCE_"
	nuc = "NN"
	if child._nuclearity == "Satellite":
		nuc = "SN" if parent._childs[0] == child else "NS"
	action += nuc
	action += "_"
	action += map_to_cluster(child._relation)
	return action
		
def gen_state(tree, stack, queue, depth):
	"""
		Generate the state of the queue and the stack
		Parameters: depth is the stack depth
	"""
	edus_idx = [0] * (depth + 1)
	edus_spans = [(0,0)] * (depth + 1)
	sents_spans = [(0,0)] * (depth + 1)
	if len(queue) > 0:
		edus_idx[depth] = queue[-1]
		edus_spans[depth] = queue[-1], queue[-1]
		sent_ind = tree._edu_to_sent_ind[queue[-1]]
		sents_spans[depth] = sent_ind, sent_ind

	act_depth = min(len(stack), depth) # actual depth
	for i in range(1, act_depth + 1):
		edus_idx[i - 1] = get_nuclear_edu_ind(stack[-i]) 
		s, t = stack[-i].get_span() # the span boundaries
		edus_spans[i - 1] = s, t
		sents_spans[i - 1] = tree._edu_to_sent_ind[s], tree._edu_to_sent_ind[t]

	return edus_idx, edus_spans, sents_spans

def get_nuclear_edu_ind(node):
	if node._type == "leaf":
		return node._span[0]
	l = node._childs[0]
	r = node._childs[1]
	if l._nuclearity == "Nucleus":
		return get_nuclear_edu_ind(l)
	return get_nuclear_edu_ind(r)


