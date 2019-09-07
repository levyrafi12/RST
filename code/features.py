from train_data import Sample
from relations_inventory import action_to_ind_map
from vocabulary import split_edu_to_tags
from vocabulary import split_edu_to_tokens
from vocabulary import vocab_get
from vocabulary import DEFAULT_TOKEN # empty string
from vocabulary import get_tag_ind
from vocabulary import gen_one_hot_vector
from vocabulary import gen_bag_of_words
from vocabulary import gen_tag_one_hot_vector

import numpy as np
import torch

import random

import sys
sys.stdout.flush()

def extract_features(trees, samples, vocab, subset_size, tag_to_ind_map, \
	bag_of_words=False, basic_feat=True, word_encoding='embedd'):
	"""
		Return a subset of samples selected in a random fasion
	"""
	x_vecs = []
	y_labels = []

	rand_samples = np.arange(len(samples))
	np.random.shuffle(rand_samples)

	for i in range(subset_size):
		sample_ind = rand_samples[i]
		extend_features_vec(samples, sample_ind, vocab, tag_to_ind_map, \
			x_vecs, y_labels, bag_of_words, basic_feat, word_encoding)

	return [x_vecs, y_labels]


def extract_features_next_subset(trees, samples, vocab, subset_size, tag_to_ind_map, \
	bag_of_words=False, basic_feat=True, word_encoding='embedd'):
	"""
		Return the next subset
	"""
	n_samples = len(samples)
	x_vecs = []
	y_labels = []

	rand_samples = np.arange(len(samples))
	np.random.shuffle(rand_samples)

	for i in range(1, n_samples + 1):
		sample_ind = rand_samples[i - 1]
		extend_features_vec(samples, sample_ind, vocab, tag_to_ind_map, \
			x_vecs, y_labels, bag_of_words, basic_feat, word_encoding)
		if i % subset_size == 0 or i == n_samples:
			yield [x_vecs, y_labels]
			x_vecs = []
			y_labels = []

def extract_sents_features(sents, pos_tags, vocab, tag_to_ind_map):
	sents_ind_emb = [] # pairs of sent ind and sentence embedding
	sent_ind = 1

	for sent, sent_pos_tags in list(zip(sents, pos_tags)):
		sent_emb = []
		words_emb = gen_words_emb(sent, vocab)
		tags_emb = gen_pos_tags_emb(sent_pos_tags, tag_to_ind_map)
		for word_e, tag_e in list(zip(words_emb, tags_emb)):
			sent_emb.append(word_e + tag_e) # concatenation
		# tensor shape is sent_len * word_emb_dim
		sents_ind_emb.append((sent_ind, torch.tensor(sent_emb)))   
		sent_ind += 1

	# sort by the number of the rows of the tensors
	sents_ind_emb.sort(key=lambda elem: len(elem[1]), reverse=True) 
	sent_ind_map = [elem[0] for elem in sents_ind_emb]
	sents_emb = [elem[1] for elem in sents_ind_emb]

	return sents_emb, sent_ind_map

def gen_words_emb(sent, vocab, use_def=False):
	vecs = []
	for word in sent:
		vec = gen_word_vectorized_feat(vocab, word, use_def, 'embedd')
		vecs.append(vec)
	return vecs

def gen_pos_tags_emb(pos_tags, tag_to_ind_map, use_def=False):
	vecs = []
	for tag in pos_tags:
		vec = gen_tag_one_hot_vector(tag_to_ind_map, tag, use_def)
		vecs.append(vec)
	return vecs

def extend_features_vec(samples, sample_ind, vocab, tag_to_ind_map, x_vecs, y_labels, \
	bag_of_words, basic_feat, word_encoding):
	_, vec_feats = add_features_per_sample(samples[sample_ind], vocab, tag_to_ind_map, \
		False, bag_of_words, basic_feat, word_encoding)
	x_vecs.append(vec_feats)
	y_labels.append(action_to_ind_map[samples[sample_ind]._action])

def add_features_per_sample(sample, vocab, tag_to_ind_map, use_def=False, \
	bag_of_words=False, basic_feat=True, word_encoding='embedd'):
	features = {}
	feat_names = []
	split_edus = []
	tags_edus = []
	head_set_edus = []
	tree = sample._tree
	for i in range(len(sample._state)):
		edu_ind = sample._state[i]
		if edu_ind > 0:
			split_edus.append(split_edu_to_tokens(tree, edu_ind))
			tags_edus.append(split_edu_to_tags(tree, edu_ind))
			head_set_edus.append(tree._EDU_head_set[edu_ind])
		else:
 			split_edus.append([''])
 			tags_edus.append([''])
 			head_set_edus.append([''])

	feat_names.append(['BEG-WORD-STACK1', 'BEG-WORD-STACK2', 'BEG-WORD-QUEUE1'])
	feat_names.append(['SEC-WORD-STACK1', 'SEC-WORD-STACK2', 'SEC-WORD-QUEUE1'])
	feat_names.append(['THIR-WORD-STACK1', 'THIR-WORD-STACK2', 'THIR-WORD-QUEUE1'])

	feat_names.append(['BEG-TAG-STACK1', 'BEG-TAG-STACK2', 'BEG-TAG-QUEUE1'])
	feat_names.append(['SEC-TAG-STACK1', 'SEC-TAG-STACK2', 'SEC-TAG-QUEUE1'])
	feat_names.append(['THIR-TAG-STACK1', 'THIR-TAG-STACK2', 'THIR-TAG-QUEUE1'])

	if basic_feat:
		for i in range(0,3):
			add_word_features(features, split_edus, feat_names[i], i)

		for i in range(0,3):
			add_tag_features(features, tags_edus, feat_names[i + 3], i, tag_to_ind_map)

		feat_names = ['END-WORD-STACK1', 'END-WORD-STACK2', 'END-WORD-QUEUE1']
		add_word_features(features, split_edus, feat_names, -1)

		feat_names = ['END-TAG-STACK1', 'END-TAG-STACK2', 'END-TAG-QUEUE1']
		add_tag_features(features, tags_edus, feat_names, -1, tag_to_ind_map)

		feat_names = ['1-HEAD-SET-STACK1', '1-HEAD-SET-STACK2', '1-HEAD-SET-QUEUE1']
		add_head_set(features, head_set_edus, feat_names, 0)

	add_edu_features(features, tree, sample._state, split_edus, vocab, \
		use_def, bag_of_words)
	vecs = gen_vectorized_features(features, vocab, tag_to_ind_map, \
		use_def, basic_feat, word_encoding)
	return features, vecs

def add_word_features(features, split_edus, feat_names, word_loc):
	for i in range(len(split_edus)):
		words = split_edus[i]
		feat = feat_names[i]
		features[feat] = DEFAULT_TOKEN
		if words != ['']:
			# last word or one of the first 3 words
			if word_loc < 0 or len(words) > word_loc:
				features[feat] = words[word_loc]

def add_tag_features(features, tags_edus, feat_names, tag_loc, tag_to_ind_map):
	for i in range(len(tags_edus)):
		tags = tags_edus[i]
		feat = feat_names[i]
		features[feat] = ''
		if tags != ['']:
			if tag_loc < 0 or len(tags) > tag_loc:
				features[feat] = tags[tag_loc]

def add_head_set(features, head_set_edus, feat_names, word_loc):
	for i in range(len(head_set_edus)):
		head_set = head_set_edus[i]
		feat = feat_names[i]
		features[feat] = DEFAULT_TOKEN
		if head_set != ['']:
			if len(head_set) > word_loc:
				features[feat] = head_set[word_loc]

def add_edu_features(features, tree, edus_ind, split_edus, vocab, use_def, \
	bag_of_words):
	feat_names = ['LEN-STACK1', 'LEN-STACK2', 'LEN-QUEUE1']

	num_edus = tree._root._span[1] - tree._root._span[0] + 1

	for i in range(0,3):
		feat = feat_names[i]
		if edus_ind[i] > 0:
			features[feat] = normalized(len(split_edus[i]), num_edus)
		else:
			features[feat] = 0 

	edu_ind_in_tree = []

	for i in range(0,3):
		if edus_ind[i] > 0:
			edu_ind_in_tree.append(edus_ind[i]) 
		else:
			edu_ind_in_tree.append(0)

	max_dist = num_edus - 1

	features['DIST-FROM-START-QUEUE1'] = normalized((edu_ind_in_tree[2] - 1), max_dist)

	features['DIST-FROM-END-STACK1'] = \
		normalized((tree._root._span[1] - edu_ind_in_tree[0]), max_dist)

	features['DIST-STACK1-QUEUE1'] = \
		normalized((edu_ind_in_tree[2] - edu_ind_in_tree[0]), max_dist)

	same_sent = tree._edu_to_sent_ind[edus_ind[0]] == \
		tree._edu_to_sent_ind[edus_ind[2]]

	features['SAME-SENT-STACK1-QUEUE1'] = 1 if same_sent else 0

	if bag_of_words:
		feat_names = ['EDU-STACK1', 'EDU-STACK2', 'EDU-QUEUE']
		for i in range(0,3):
			feat = feat_names[i]
			features[feat] = gen_bag_of_words(tree, vocab, edus_ind[i], use_def)

def gen_vectorized_features(features, vocab, tag_to_ind_map, use_def, basic_feat, \
	word_encoding):
	vecs = []
	n_tags = len(tag_to_ind_map) - 1
	for key, val in features.items():
		if not basic_feat and 'EDU' not in key:
			continue
		# print("key {} val '{}'".format(key, val))
		if 'HEAD-SET' in key:
			# overwrite use_def to True since vocab and dependency graph were 
			# built by different tokenizer. Thus some head word set may not exist in 
			# vocab even in training data set.
			vecs += gen_word_vectorized_feat(vocab, val, True, word_encoding)
		elif 'WORD' in key:
			vecs += gen_word_vectorized_feat(vocab, val, use_def, word_encoding)
		elif 'TAG' in key:
			# vecs += [normalized(get_tag_ind(tag_to_ind_map, val, use_def), n_tags)]
			vecs += gen_tag_one_hot_vector(tag_to_ind_map, val, use_def)
		elif 'EDU' in key:
			vecs += val
		else:
			vecs += [val]
		# print(len(vecs))
	return vecs

def normalized(val, max_val):
	return val / max_val

def gen_word_vectorized_feat(vocab, val, use_def, word_encoding):
	word_ind = vocab_get(vocab, val, use_def)
	if word_encoding == 'embedd':
		vec = [elem for elem in vocab._wordVectors[word_ind]]
	else:
		vec = gen_one_hot_vector(vocab, word_ind)
	return vec

def project_features(A, x_vecs):
	"""
		Projectiong each x_vec in x_vecs from v-dim space to k dim space  
		A is a matrix of dimension k * v
	"""
	v = np.array(x_vecs).T # v * n
	Av = np.zeros((A.shape[0], v.shape[1]))

	for i in range(A.shape[0]):
		for j in range(v.shape[1]):
			Av[i,j] = np.matmul(A[i, :], v[:, j])

	return Av.T # n * v

def get_word_encoding(model_name):
	if model_name == 'neural':
		return 'embedd'
	return 'one_hot'

def is_bag_of_words(model_name):
	return model_name == 'dplp_A_I' or model_name == 'dplp'

def is_basic_feat(model_name):
	return model_name != 'dplp'