import glob
import random
import numpy as np
import torch
import os

from preprocess import Node
from preprocess import print_serial_file
from preprocess import extract_base_name_file
from evaluation import eval
from features import add_features_per_sample
from train_data import Sample
from train_data import gen_state
from model import neural_net_predict
from model import linear_predict
from relations_inventory import ind_to_action_map
from preprocess import create_dir
from preprocess import build_infile_name
from preprocess import SEP

class Stack(object):
	def __init__(self):
		self._stack = []

	def pop(self):
		return self._stack.pop(-1)

	def push(self, elem):
		return self._stack.append(elem)

	def size(self):
		return len(self._stack)

class Queue(object):
	def __init__(self):
		self._EDUS = []

	@classmethod
	def read_file(cls, filename):
		# print("{}".format(filename))
		queue = Queue()
		with open(filename) as fh:
			for line in fh:
				line = line.strip()
				queue._EDUS.append(line)
			queue._EDUS[::-1]
		return queue

	def empty(self):
		return self._EDUS == []

	def pop(self):
		return self._EDUS.pop(-1)

	def len(self):
		return len(self._EDUS)

class Transition(object):
	def __init__(self):
		self._nuclearity = [] # <nuc>, <nuc>
		self._relation = '' # cluster relation
		self._action = '' # shift or 'reduce'

	def gen_str(self):
		s = self._action
		if s != 'shift':
			s += "-"
			s += ''.join([elem[0] for elem in self._nuclearity])
			s += "-"
			s += self._relation
		return s.upper()

def parse_files(base_path, model_name, model, trees, vocab, \
	y_all, tag_to_ind_map, baseline, infiles_dir, gold_files_dir, pred_outdir="pred"):
	path_to_out = create_dir(base_path, pred_outdir)

	for tree in trees: 
		fn = build_infile_name(tree._fname, base_path, infiles_dir, ["out.edus", "edus"])
		queue = Queue.read_file(fn)
		stack = Stack()
		root = parse_file(queue, stack, model_name, model, tree, \
			vocab, y_all, tag_to_ind_map, baseline)
		predfn = path_to_out
		predfn += SEP
		predfn += tree._fname
		with open(predfn, "w") as ofh:
			print_serial_file(ofh, root, False)

	eval(gold_files_dir, "pred")

def parse_file(queue, stack, model_name, model, tree, \
	vocab, y_all, tag_to_ind_map, baseline):

	leaf_ind = 1
	while not queue.empty() or stack.size() != 1:
		node = Node()
		node._relation = 'SPAN'

		if baseline:
			transition = most_freq_baseline(queue, stack)
		else:
			transition = predict_transition(queue, stack, model_name, model, \
				tree, vocab, y_all, tag_to_ind_map, leaf_ind)

		# print("queue size = {} , stack size = {} , action = {}".\
		# format(queue.len(), stack.size(), transition.gen_str()))

		if transition._action == "shift":
			# create a leaf
			node._text = queue.pop()
			node._type = 'leaf'
			node._span = [leaf_ind, leaf_ind]
			leaf_ind += 1
		else:
			r = stack.pop()
			l = stack.pop()
			node._childs.append(l)
			node._childs.append(r)
			l._nuclearity = transition._nuclearity[0]
			r._nuclearity = transition._nuclearity[1]
			if l._nuclearity == "Satellite":
				l._relation = transition._relation
			elif r._nuclearity == "Satellite":
				r._relation = transition._relation	
			else:
				l._relation = transition._relation
				r._relation = transition._relation

			if queue.empty() and stack.size() == 0:
				node._type = "Root"
			else:
				node._type = "span"
			node._span = [l._span[0], r._span[1]]
		stack.push(node)

	return stack.pop()

def predict_transition(queue, stack, model_name, model, tree, vocab, \
	y_all, tag_to_ind_map, top_ind_in_queue):
	transition = Transition()

	sample = Sample()
	sample._state = gen_config(queue, stack, top_ind_in_queue)
	sample._tree = tree
	# sample.print_info()

	_, x_vecs = add_features_per_sample(sample, vocab, tag_to_ind_map, True)

	if model_name == "neural":
		pred = neural_net_predict(model, x_vecs)
		action = ind_to_action_map[pred.argmax()]
		_, indices = torch.sort(pred)
		# scores = [(vals[i], ind_to_action_map[indices[i]]) for i in reversed(range(len(pred)))]
		# print("scores {} \n best {} {}".format(scores[:15], pred.argmax(), action))
	else:
		pred = linear_predict(model, [x_vecs])
		action = ind_to_action_map[y_all[np.argmax(pred)]]
		indices = np.argsort(pred)	

	# correct invalid action
	if stack.size() < 2 and action != "SHIFT":
		action = "SHIFT"
	elif queue.len() <= 0 and action == "SHIFT":
		action = ind_to_action_map[indices[-2]]

	if action == "SHIFT":
		transition._action = "shift"	
	else:	 
		transition._action = "reduce"

		split_action = action.split("-")
		nuc = split_action[1]
		rel = split_action[2]

		if nuc == "NS":
			transition._nuclearity.append("Nucleus")
			transition._nuclearity.append("Satellite")
		elif nuc == "SN":
			transition._nuclearity.append("Satellite")
			transition._nuclearity.append("Nucleus")
		else:
			transition._nuclearity.append("Nucleus")
			transition._nuclearity.append("Nucleus")
		transition._relation = rel

	return transition

def gen_config(queue, stack, top_ind_in_queue):
	q_temp = []
	if queue.len() > 0: # queue contains element texts not indexes
		q_temp.append(top_ind_in_queue)

	return gen_state(stack._stack, q_temp)

def most_freq_baseline(queue, stack):
	transition = Transition()

	if stack.size() < 2:
		transition._action = "shift"
	elif not queue.empty():
		actions = ["shift", "reduce"]
		ind = random.randint(0,1)
		transition._action = actions[ind]
	else:
		transition._action = "reduce"
		
	if transition._action == "shift":
		return transition

	transition._relation = 'ELABORATION'
	transition._nuclearity.append("Nucleus")
	transition._nuclearity.append("Satellite")

	return transition