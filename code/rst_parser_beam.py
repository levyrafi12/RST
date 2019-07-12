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
import copy

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

class Buffer(object):
	def __init__(self):
		self._EDUS = []

	@classmethod
	def read_file(cls, filename):
		# print("{}".format(filename))
		buffer = Buffer()
		with open(filename) as fh:
			for line in fh:
				line = line.strip()
				buffer._EDUS.append(line)
			buffer._EDUS[::-1]
		return buffer

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
		self._score = 1 # log scale

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
		self._buffer = Buffer()
		self._stack = Stack()
		self._transitions = []
		self._score = 1 # path score in log scale
		self._root = ''
		# index of EDU at the front of the buffer
		self._leaf_ind = 1 
		self._level = 0

	def read_file(fn):
		self._buffer = Buffer.read_file(fn)

	def ended(self):
		return self._buffer.empty() and self._stack.size() == 1

	def gen_str(self):
		s = self._transitions[-1].gen_str()
		s += " , leaf ind "
		s += str(self._leaf_ind)
		s += " , path score "
		s += str(self._score)
		return s

class ParsersQueue(object):
	def __init__(self):
		self._parsers = []

	def __init__(self, fn, k_top):
		parser = Parser()
		parser.read_file(fn)
		self._parsers = [parser]
		self._ended_parsers = []
		self._k_top = k_top

	def func_key(parser):
		return parser._score

	def reduce(self):
		self._parsers = sorted(self._parsers, key=func_key)
		self._parsers = self._parsers[::-1]
		self._parsers = self._parsers[:self._k_top]

	def ended(self):
		return self._parsers[0].ended()

	def pop(self):
		return self._parsers.pop(-1)

def parse_files(base_path, model_name, model, trees, vocab, \
	y_all, tag_to_ind_map, baseline, infiles_dir, \
	k_top, pred_oudir="pred"):
	path_to_out = create_dir(base_path, pred_outdir)

	for tree in trees: 
		fn = build_infile_name(tree._fname, base_path, infiles_dir, ["out.edus", "edus"])
		root = parse_file(fn, model_name, model, tree, vocab, max_edus, \
			y_all, tag_to_ind_map, k_top)
		predfn = path_to_out
		predfn += SEP
		predfn += tree._fname
		with open(predfn, "w") as ofh:
			print_serial_file(ofh, root, False)

	eval(gold_files_dir, "pred")

def parse_file(fn, model_name, model, tree, vocab, \
	y_all, tag_to_ind_map, baseline, k_top):
	parsers_queue = ParsersQueue(fn, k_top)
	level = 0

	while not parsers_queue.empty():
		parser = parsers_queue.top()

		if (parser._level > level):
			parsers_queue.reduce()
			level += 1
			continue

		parser = parsers_queue.pop()
		# transition = most_freq_baseline(queue, stack)
		next_move(parsers_queue, parser, model_name, model, tree, vocab, \
			y_all, tag_to_ind_map)

	return parsers_queue.best_parser()._root

def next_transitions(parsers_queue, parser, model_name, model, tree, vocab, \
	y_all, tag_to_ind_map, top_ind_in_queue, k_top):

	sample = Sample()
	sample._state = gen_config(parser._queue, parser._stack, top_ind_in_queue)
	sample._tree = tree

	# sample.print_info()

	_, x_vecs = add_features_per_sample(sample, vocab, \
		tag_to_ind_map, True)

	if model_name == "neural":
		scores, indices, actions = neural_net_predict(model, x_vecs)
	else:
		scores, indices, actions = linear_predict(model, [x_vecs], y_all)

	# print("{}".format(actions[0:parsers_queue._k_top]))
	# print ("{}".format(scores[0:parsers_queue._k_top]))

	i = 0
	done = False

	while not done:
		action = actions[i]
		i += 1

		# illegal actions
		if queue.len() <= 0 and action == "SHIFT":
			continue

		if stack.size() < 2 and action != "SHIFT":
			action = "SHIFT"

		if stack.size() < 2 or i >= k_top:
			done = True

		# print("action {} queue len() {}".format(action, \
		# 	parser._buffer.len()))

		transition = set_transition(action)
		
		if k_top > 1:
			parsers_queue.push(copy.copy(parser))

		apply_transition(parsers_queue.back(), transition)

def gen_config(queue, stack, top_ind_in_queue):
	q_temp = []
	if queue.len() > 0: # queue contains element texts not indexes
		q_temp.append(top_ind_in_queue)

	return gen_state(stack._stack, q_temp)

def gen_transition(action):
	transition = Transition()

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

def apply_transition(parser, transition):
	node = Node()
	node._relation = 'SPAN'
	if transition._action == "shift":
		# create a leaf
		node._text = queue.pop()
		node._type = 'leaf'
		node._span = [leaf_ind, leaf_ind]
		parser._leaf_ind += 1
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

		if parser._buffer.empty() and parser._stack.size() == 0:
			node._type = "Root"
		else:
			node._type = "span"
		node._span = [l._span[0], r._span[1]]
	parser._stack.push(node)

	# print("buffer size = {} , stack size = {} , action = {}".\
	# format(parser._buffer.len(), parser._stack.size(), transition.gen_str()))

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
