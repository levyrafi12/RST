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
from preprocess import remove_dir
from collections import deque
import copy

correct_illegal_action = True
print_transition = True

OUTDIR = "pred"

class Stack(object):
	def __init__(self):
		self._stack = []

	def top(self):
		return self._stack[-1]

	def pop(self):
		return self._stack.pop(-1)

	def push(self, elem):
		return self._stack.append(elem)

	def size(self):
		return len(self._stack)

	def copy(self):
		return copy.copy(self._stack)

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

	def copy(self):
		return copy.copy(self._EDUS)

class Transition(object):
	def __init__(self):
		self._nuclearity = [] # <nuc>, <nuc>
		self._relation = '' # cluster relation
		self._action = '' # shift or 'reduce'
		self._score = 0

	def gen_str(self):
		s = self._action
		if s != 'shift':
			s += "-"
			s += ''.join([elem[0] for elem in self._nuclearity])
			s += "-"
			s += self._relation
		s += " , score "
		s += str(self._score)
		return s.upper()

class Parser(object):
	def __init__(self):
		self._queue = []
		self._stack = Stack()
		self._transitions = []
		self._score = 0
		self._root = ''
		self._leaf_ind = 1 # the EDU at the top of the queue
		self._path_id = "1"

	def completed(self):
		return self._queue.empty() and self._stack.size() == 1

	def gen_str(self):
		s = self._transitions[-1].gen_str()
		s += " , leaf ind "
		s += str(self._leaf_ind)
		s += " , toal score "
		s += str(self._score)
		s += " , path id "
		s += self._path_id
		return s

def parse_files(base_path, gold_files_dir, model_name, model, trees, vocab, \
	max_edus, y_all, tag_to_ind_map, infiles_dir, k_top):
	path = base_path

	remove_dir(base_path, OUTDIR)
	path_to_out = base_path
	path_to_out += "\\"
	path_to_out += OUTDIR
	os.makedirs(path_to_out)

	for tree in trees: 
		fn = base_path
		fn += "\\"
		fn += infiles_dir
		fn += "\\"
		fn += tree._fname
		fn += ".out.edus"
		print("Parsing tree {}".format(tree._fname))
		root = parse_file(fn, model_name, model, tree, vocab, max_edus, \
			y_all, tag_to_ind_map, k_top)
		predfn = path_to_out
		predfn += "\\"
		predfn += tree._fname
		with open(predfn, "w") as ofh:
			print_serial_file(ofh, root, False)

	eval(gold_files_dir, "pred")
		# n1 = count_lines(predfn) 
		# n2 = count_lines(goldfn)
		# print("{} {} {} {} equal: {}".format(predfn, n1, goldfn, n2, n1 == n2))

def parse_file(fn, model_name, model, tree, vocab, max_edus, y_all, tag_to_ind_map, k_top):
	best_score = 0
	best_root = None

	parsers = deque()
	parsers.append(Parser())
	parsers[0]._queue = Queue.read_file(fn)

	while parsers:
		parser = parsers.pop()
		if parser.completed():
			print("completed {}".format(parser.gen_str()))
			if best_score < parser._score:
				best_score = parser._score
				best_root = parser._root
			continue

		# transition = most_freq_baseline(queue, stack)
		next_transitions = predict_transitions(parser._queue, parser._stack, model_name, model, \
			tree, vocab, max_edus, y_all, tag_to_ind_map, parser._leaf_ind, k_top)

		for i in range(len(next_transitions)):
			next_transition = next_transitions[i]
			next_parser = Parser()
			next_parser._queue = Queue.read_file(fn)
			next_parser._score = parser._score + next_transition._score
			next_parser._transitions = copy.copy(parser._transitions)
			next_parser._transitions.append(next_transition)
			next_parser._path_id = parser._path_id
			next_parser._path_id += ":"
			next_parser._path_id += str(i + 1)

			leaf_ind = 1
			for transition in next_parser._transitions:
				leaf_ind = apply_transition(transition, next_parser._queue, \
					next_parser._stack, model_name, model, tree, vocab, max_edus, \
					y_all, leaf_ind, tag_to_ind_map)

			if next_parser._stack.size() > 0:
				next_parser._root = next_parser._stack.top()
			next_parser._leaf_ind = leaf_ind
			parsers.appendleft(next_parser)

	return best_root

def apply_transition(transition, queue, stack, model_name, model, tree, \
	vocab, max_edus, y_all, leaf_ind, tag_to_ind_map):
	if print_transition:
		print("queue size = {} , stack size = {} , action = {}".\
		format(queue.len(), stack.size(), transition.gen_str()))

	node = Node()
	node._relation = 'SPAN'
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

	return leaf_ind

def predict_transitions(queue, stack, model_name, model, tree, vocab, \
	max_edus, y_all, tag_to_ind_map, top_ind_in_queue, k_top):

	sample = Sample()
	sample._state = gen_config(queue, stack, top_ind_in_queue)
	sample._tree = tree
	sample.print_info()

	_, x_vecs = add_features_per_sample(sample, vocab, max_edus, 
		tag_to_ind_map, True)

	if model_name == "neural":
		scores, indices = neural_net_predict(model, x_vecs)
		scores_actions = [(scores[i], ind_to_action_map[indices[i]]) for i in range(len(indices))]
		print("{}".format(scores_actions[0:15]))
	else:
		scores, indices = linear_predict(model, [x_vecs])

	transitions = []

	print("k_top {}".format(k_top))

	for i in range(k_top):
		if model_name == "neural":
			action = ind_to_action_map[indices[i]]
		else:
			action = ind_to_action_map[y_all[indices[i]]]
		
		# illegal actions
		if stack.size() < 2 and action != "SHIFT":
			action = "SHIFT"

		elif queue.len() <= 0 and action == "SHIFT":
			if k_top > 1:
				continue
			else:
				action = ind_to_action_map[indices[-2]]

		print("action {} queue len() {}".format(action, queue.len()))

		transition = Transition()
		transition._score = scores[i]
		transitions.append(transition)

		if action == "SHIFT":
			transition._action = "shift"
			if stack.size() < 2:
				break
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

	return transitions

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

def count_lines(filename):
    lines = 0
    for line in open(filename):
        lines += 1
    return lines

if __name__ == '__main__':
	parse_files("..")
