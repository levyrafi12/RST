from preprocess import Node
from preprocess import TreeInfo
from utils import map_to_cluster
from glove import loadWordVectors
from relations_inventory import action_to_ind_map
from preprocess import build_file_name
from preprocess import SEP
from preprocess import DEFAULT_TOKEN
from preprocess import DEFAULT_TOKEN2

import re
import glob
import copy
import numpy as np
import os

DEFAULT_TAG = ''

class Vocab(object):
	def __init__(self):
		self._tokens = { DEFAULT_TOKEN: 0, DEFAULT_TOKEN2 : 1} 
		self._wordVectors = []

	def len(self):
		return len(self._tokens)
		
def gen_vocabulary(trees, base_path, files_dir="TRAINING", glove_dir="glove", \
	glove_dim=50, print_vocab=False):
	vocab = Vocab()

	word_ind = 2
	for tree in trees:
		for word_list in tree._edu_tokenized_table[1:]:
			for word in word_list:
				if not vocab._tokens.get(word.lower()):
					vocab_set(vocab, word, word_ind)
					word_ind += 1
		# tokenizing the sentence may slightly produce different results 
		# than tokenizing the EDUS of the sentence.
		for sent in tree._sent_tokenized_table[1:]:
			for word in sent:
				if not vocab._tokens.get(word.lower()):
					vocab_set(vocab, word, word_ind)
					word_ind += 1

	glove_fn = base_path
	glove_fn += SEP
	glove_fn += glove_dir
	glove_fn += SEP
	glove_fn += "glove.6B."
	glove_fn += str(glove_dim)
	glove_fn += "d.txt"

	assert os.path.exists(glove_fn), "file does not exists: " + glove_fn
	
	vocab._wordVectors = loadWordVectors(vocab._tokens, glove_fn, glove_dim)

	if print_vocab:
		n_founds = 0
		for key, val in vocab._tokens.items():
			found = False
			# a token has a zeros vector if that token did not exist in glove   
			if list(vocab._wordVectors[val]).count(0) < len(vocab._wordVectors[val]):
				found = True
				n_founds += 1
			# print("key = {} ind = {} in dict = {}".format(key, val, found))

		# print("words in dictionary {}%".format(n_founds / len(vocab._tokens) * 100))

	tag_to_ind_map = build_tags_dict(trees)

	return [vocab, tag_to_ind_map]

def split_edu_to_tokens(tree, edu_ind):
	return tree._edu_tokenized_table[edu_ind]

def split_edu_to_tags(tree, edu_ind):
	return tree._edu_pos_tags_table[edu_ind]

def gen_one_hot_vector(vocab, ind):
	"""
		generate one hot vector for a token ind (word)
	""" 
	vec = len(vocab._tokens) * [0]
	vec[ind] = 1
	return vec

def gen_tag_one_hot_vector(tag_to_ind_map, val, use_def_word=False):
	"""
		generate one hot vector for a token ind (word)
	""" 
	vec = len(tag_to_ind_map) * [0]
	tag_ind = get_tag_ind(tag_to_ind_map, val, use_def_word)
	vec[tag_ind] = 1
	return vec

def gen_bag_of_words(tree, vocab, edu_ind, use_def_word=False):
	zeros = len(vocab._tokens) * [0]
	if edu_ind == 0:
		return zeros

	vec = zeros
	tokens = split_edu_to_tokens(tree, edu_ind)
	for token in tokens:
		ind = vocab_get(vocab, token, use_def_word)
		vec[ind] += 1
	return vec	

def get_tag_ind(tag_to_ind_map, tag, use_def_tag=False):
	if tag_to_ind_map.get(tag, None) == None:
		if not use_def_tag:
			assert False, "Could not find tag: " + tag
		return tag_to_ind_map[''] # empty string is treated as the default tag
	return tag_to_ind_map[tag]

def build_tags_dict(trees):
	tag_to_ind_map = {DEFAULT_TAG: 0}
	tag_ind = 1

	for tree in trees:
		for tag_list in tree._edu_pos_tags_table[1:]:
			tag_ind = add_tag_list_to_dict(tag_list, tag_ind, tag_to_ind_map)
		for tag_list in tree._sent_pos_tags_table[1:]:
			tag_ind = add_tag_list_to_dict(tag_list, tag_ind, tag_to_ind_map)

	return tag_to_ind_map

def add_tag_list_to_dict(tag_list, tag_ind, tag_to_ind_map):
	for tag in tag_list:
		if tag_to_ind_map.get(tag, None) == None:
			tag_to_ind_map[tag] = tag_ind
			tag_ind += 1
	return tag_ind

def vocab_get(vocab, word, use_def_word=False, def_word=DEFAULT_TOKEN):
	val = vocab._tokens.get(word.lower(), None)
	if val != None:
		return val

	if use_def_word:
		return vocab._tokens.get(def_word)

	assert False, "word not in vocabulary: " + word.lower()

def vocab_set(vocab, word, ind):
	vocab._tokens[word.lower()] = ind
