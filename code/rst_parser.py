import glob
import random
import numpy as np
import torch
import os
import sys

sys.stdout.flush()

from preprocess import Node
from preprocess import print_serial_file
from preprocess import extract_base_name_file
from preprocess import preprocess
from evaluation import eval
from features import add_features_per_sample
from train_data import Sample
from train_data import gen_state
from predict import predict
from features import get_word_encoding
from features import is_bag_of_words
from relations_inventory import ind_to_action_map
from relations_inventory import action_to_ind_map
from preprocess import create_dir
from preprocess import build_infile_name
from preprocess import SEP
from defs import *
import copy
from collections import deque
from collections import defaultdict
import math

class Stack(object):
	def __init__(self):
		self._stack = []

	def __copy__(self):
		other = Stack()
		other._stack = copy.copy(self._stack)
		return other

	def top(self):
		return self._stack[-1]

	def pop(self):
		return self._stack.pop(-1)

	def push(self, elem):
		return self._stack.append(elem)

	def size(self):
		return len(self._stack)

class Buffer(object):
	def __init__(self):
		self._EDUS = []

	def __copy__(self):
		other = Buffer()
		other._EDUS = copy.copy(self._EDUS)
		return other

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

class Parser(object):
	def __init__(self):
		self._buffer = Buffer()
		self._stack = Stack()
		self._score = 0
		self._root = ''
		# index of EDU at the front of the buffer
		self._leaf_ind = 1 
		self._level = 0

	def __copy__(self):
		other = Parser()
		other._buffer = copy.copy(self._buffer)
		other._stack = copy.copy(self._stack)
		other._score = self._score
		other._root = self._root
		other._leaf_ind = self._leaf_ind
		other._level = self._level
		return other

	def read_file(self, fn):
		self._buffer = Buffer.read_file(fn)

	def ended(self):
		return self._buffer.empty() and self._stack.size() == 1

class ParsersQueue(object):
	def __init__(self):
		self._parsers = deque()

	def __init__(self, fn, k_top):
		parser = Parser()
		parser.read_file(fn)
		self._parsers = deque([parser])
		self._k_top = k_top

	def len(self):
		return len(self._parsers) 

	def reduce(self):
		self._parsers = deque(sorted(self._parsers, reverse=True, key=lambda x : x._score))
		i = len(self._parsers)
		while i > self._k_top:
			i -= 1
			parser = self._parsers.pop() # keep the best k parsers only
			del(parser)

	def pop_front(self):
		return self._parsers.pop()

	def front(self):
		return self._parsers[-1]

	def back(self):
		return self._parsers[0]

	def push_back(self, parser):
		self._parsers.appendleft(parser)

def evaluate(model_name, model, vocab, tag_to_ind_map, baseline=False, k_top=1):
	dev_trees = preprocess(WORK_DIR, DEV_TEST_DIR, DEV_TEST_GOLD_DIR)
	parse_files(model_name, model, dev_trees, vocab, tag_to_ind_map, baseline, k_top)

def parse_files(model_name, model, trees, vocab, tag_to_ind_map, baseline, k_top):
	path_to_out = create_dir(WORK_DIR, PRED_OUTDIR)

	for tree in trees: 
		fn = build_infile_name(tree._fname, WORK_DIR, DEV_TEST_DIR, ["out.edus", "edus"])
		root = parse_file(fn, model_name, model, tree, vocab, \
			tag_to_ind_map, baseline, k_top)
		predfn = path_to_out
		predfn += SEP
		predfn += tree._fname
		with open(predfn, "w") as ofh:
			print_serial_file(ofh, root, False)

	eval(DEV_TEST_GOLD_DIR, "pred")

def parse_file(fn, model_name, model, tree, vocab, \
	tag_to_ind_map, baseline, k_top):
	parsers_queue = ParsersQueue(fn, k_top)
	# N shift operations + N - 1 reduce relations
	max_level = 2 * parsers_queue.back()._buffer.len() - 1 
	wrong_decisions = defaultdict(int)
	level = 0
	
	while level < max_level:
		parser = parsers_queue.front()
		if (parser._level > level):
			parsers_queue.reduce() # sort and keep the k parsers with best scores, rest are deleted
			level += 1
			continue

		parser = parsers_queue.pop_front()
		next_move(parsers_queue, parser, model_name, model, tree, vocab, \
			tag_to_ind_map, baseline, wrong_decisions)

	# make sure parsing indeed ended corretly
	assert len([x for x in parsers_queue._parsers if x.ended()]) == parsers_queue.len(), \
		"some parsers were not completed"
	assert [x for x in parsers_queue._parsers if x._level < max_level] == [], \
		"all parsers completed but some did not reach max level"
	assert parsers_queue.back()._root._type == 'Root', \
		"Bad root type"

	# print("SHIFT when buf empty: {}, REDUCE when stack size < 2: {}, num decisions: {}".\
	# format(wrong_decisions["SHIFT"], wrong_decisions["REDUCE"], max_level))

	# print("final score {0:.3f}".format(parsers_queue.back()._score))
	# parsers are sorted in descending order
	return parsers_queue.back()._root

def next_move(parsers_queue, parser, model_name, model, tree, vocab, tag_to_ind_map, \
	baseline, wrong_decisions):
	if parser.ended() or parsers_queue._k_top == 1:
		parsers_queue.push_back(parser)
		if baseline:
			transition = most_freq_baseline(parser)
			apply_transition(parser, transition)
		if parser.ended() or baseline:
			return

	sample = Sample()
	sample._state = gen_config(parser._buffer, parser._stack, parser._leaf_ind)
	sample._tree = tree

	# sample.print_info()

	_, x_vecs = add_features_per_sample(sample, vocab, tag_to_ind_map, \
		is_bag_of_words(model_name), get_word_encoding(model_name), True)

	# print("next move")
	scores, sorted_scores, sorted_actions = predict(model, model_name, x_vecs)

	i = 0
	done = False

	while not done:
		action = sorted_actions[i]
		score = sorted_scores[i]
		i += 1

		# illegal move
		if parser._buffer.len() <= 0 and action == "SHIFT":
			wrong_decisions["SHIFT"] += 1
			continue

		# fix illegal move
		if parser._stack.size() < 2 and action != "SHIFT":
			score = scores[action_to_ind_map.get("SHIFT")]
			action = "SHIFT"
			wrong_decisions["REDUCE"] += 1

		if parser._stack.size() < 2 or i >= parsers_queue._k_top:
			done = True

		transition = gen_transition(action)
		
		if parsers_queue._k_top > 1:
			parsers_queue.push_back(copy.copy(parser))

		apply_transition(parsers_queue.back(), transition)

		parsers_queue.back()._score += score
		parsers_queue.back()._level += 1

		"""
		args = "path score {0:.3f} score {1:.3f} buffer size {2}"
		args += " stack size {3} trans {4} node {5} level {6}"
		print(args.format(parsers_queue.back()._score, score, \
			parsers_queue.back()._buffer.len(), \
			parsers_queue.back()._stack.size(), \
			transition.gen_str(), parsers_queue.back()._root._type, \
			parsers_queue.back()._level))
		"""

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
		node._text = parser._buffer.pop()
		node._type = 'leaf'
		node._span = [parser._leaf_ind, parser._leaf_ind]
		parser._leaf_ind += 1
	else:
		r = parser._stack.pop()
		l = parser._stack.pop()
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
	parser._root = node

	# print("buffer size = {} , stack size = {} , action = {} node type {} level {}".\
	# format(parser._buffer.len(), parser._stack.size(), transition.gen_str(), node._type,
	# parser._level))

def most_freq_baseline(parser):
	parser._level += 1
	buffer = parser._buffer
	stack = parser._stack

	transition = Transition()

	if stack.size() < 2:
		transition._action = "shift"
	elif not buffer.empty():
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
