import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from features import gen_word_vectorized_feat
from vocabulary import gen_tag_one_hot_vector
from relations_inventory import action_to_ind_map
from general import print_memory_usage
 
from preprocess import DEFAULT_TOKEN
from preprocess import DEFAULT_TOKEN2
from vocabulary import DEFAULT_TAG

def get_samples_subset_next(samples, batch_size):
	n_samples = len(samples)
	rand_samples = np.arange(n_samples)
	np.random.shuffle(rand_samples)
	samples_subset = []

	for i in range(1, len(samples) + 1):
		rand_ind = rand_samples[i - 1]
		sample = samples[rand_ind]
		samples_subset.append(sample)
		if i % batch_size == 0 or i == n_samples:
			yield samples_subset
			samples_subset = []
		

def prepare_sents_as_inp_vecs_next(samples, vocab, tag_to_ind_map, batch_size, use_def=False):
	tuples = [] # tuples of 'batch to sent ind' and 'vectorized word_tag seq'
	n_sents = 0
	i = 1

	for sample in samples:
		for s, t in sample._sents_spans:
			n_sents += t - s + 1

	for sample in samples:
		for s, t in sample._sents_spans:
			for k in range(s, t + 1):
				words = sample._tree._sent_tokenized_table[k]
				tags = sample._tree._sent_pos_tags_table[k]
				vect_words = gen_vectorized_words(words, vocab, use_def)
				vect_tags = gen_vectorized_tags(tags, tag_to_ind_map)
				vect_words_tags = []
				for vect_word, vect_tag in list(zip(vect_words, vect_tags)):
					vect_words_tags.append(vect_word + vect_tag) # concatenation
				# tensor shape is the words_seq_len * vect_word_tag_len
				tuples.append((k, sample._tree, torch.tensor(vect_words_tags)))
				# print_memory_usage('7')  
				if i % batch_size == 0 or i == n_sents:
					# sort by the length of sents in descendent order
					# x_vecs is a list of tensors of shape sent len * word_tag_vect_len
					[batch_to_sent_ind, batch_to_tree, x_vecs] = sort_tuples_and_split(tuples, True)
					# stack a list of tensors along a new dimension (batch), and pad them 
					# to an equal length.
					# x_padded_vecs is a tensor of n_sents * max_sent_len * n_features
					x_padded_vecs = pad_sequence(x_vecs, batch_first=True)
					yield x_padded_vecs, batch_to_sent_ind, batch_to_tree
					tuples = []
				i += 1

def prepare_edus_seq_as_inp_vecs(samples, vocab, tag_to_ind_map, batch_size):
	tuples = [] # tuples of tree and seq_edus
	n_edus = 0
	i = 1

	for sample in samples:
		for s, t in sample._spans:
			n_edus += t - s + 1

	for sample in samples:
		for s, t in sample._spans:
			edus_seq = []
			for k in range(s, t + 1):
				edus_seq.append(sample._tree._edu_represent_table[k])
			# tensor dim is edus_seq_len * (2 * hidden_size)
			tuples.append((sample._tree, (s, t), torch.stack(edus_seq)))
			if i % batch_size == 0 or i == n_edus:
				# sort by the length of edus seq in descendent order
				# x_vecs is a list of tensors of shape edus seq len * (2 * hidden_size)
				[batch_to_tree, batch_to_span, x_vecs] = sort_tuples_and_split(tuples, True)
				# x_padded_vecs is a tensor of n_sents * max_sent_len * n_features
				x_padded_vecs = pad_sequence(x_vecs, batch_first=True)
				yield x_padded_vecs, batch_to_tree, batch_to_span
				tuples = []
			i += 1

def extract_edus_subtrees_hidden_repr(samples, vocab):
	x_vecs = []
	y_labels = []

	for sample in samples:
		inp_vec = extract_edus_subtrees_hidden_repr_per_sample(sample, vocab)
		inp_vec = torch.tensor(inp_vec)
		x_vecs.append(inp_vec)
		y_labels.append(action_to_ind_map[sample._action])

	y_labels = torch.tensor(y_labels, dtype=torch.long)
	# A two dimension tensor of dimension batch_size * (6 * hidden_size)
	x_vecs = torch.stack(x_vecs)
	return x_vecs, y_labels

def extract_edus_subtrees_hidden_repr_per_sample(sample, vocab):
	inp_vec = []
	for edu_i, edu_j in sample._spans:
		inp_vec += average_pooling_of_edus_repr(sample._tree, edu_i, edu_j, vocab)
	return inp_vec

def gen_vectorized_words(sent, vocab, use_def=False):
	vecs = []
	for word in sent:
		vec = gen_word_vectorized_feat(vocab, word, use_def, 'embedd')
		vecs.append(vec)
	return vecs

def gen_vectorized_tags(pos_tags, tag_to_ind_map, use_def=False):
	vecs = []
	for tag in pos_tags:
		vec = gen_tag_one_hot_vector(tag_to_ind_map, tag, use_def)
		vecs.append(vec)
	return vecs

def len_of_vectorized_word_tag(tree, vocab, tag_to_ind_map):
	word = tree._sent_tokenized_table[1][0]
	tag = tree._sent_pos_tags_table[1][0]
	[word_vec] = gen_vectorized_words([word], vocab)
	[tag_vec] = gen_vectorized_tags([tag], tag_to_ind_map)
	return len(word_vec + tag_vec)

def sort_tuples_and_split(tuples, reverse):
	result = []
	n_members = len(tuples[0])
	# assume key is the last member
	tuples.sort(key=lambda elem: len(elem[n_members - 1]), reverse=reverse) 
	for i in range(n_members):
		result.append([elem[i] for elem in tuples])
	return result

def gen_def_edu_hidden_repr(vocab):
	vec = gen_word_vectorized_feat(vocab, DEFAULT_TOKEN, False, 'embedd')
	vec += gen_word_vectorized_feat(vocab, DEFAULT_TOKEN2, False, 'embedd')
	return vec

def average_pooling_of_edus_repr(tree, edu_i, edu_j, vocab):
	"""
		Generate subtree hidden representation by average pooling
		of subtree's covering edus { edu_i, edu_i_1... edu_j}
	"""
	if edu_i == 0:
		return gen_def_edu_hidden_repr(vocab)
	subtree_repr = 0

	# print(len(tree._encoded_edu_table[1:]))
	# print("{} {}".format(edu_i, edu_j))

	for k in range(edu_i, edu_j + 1):
		subtree_repr += tree._encoded_edu_table[k]

	return (subtree_repr / (edu_j - edu_i + 1)).tolist()