import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence

import math
import numpy as np
import psutil

from features import gen_word_vectorized_feat
from vocabulary import gen_tag_one_hot_vector
from relations_inventory import action_to_ind_map

from preprocess import DEFAULT_TOKEN
from preprocess import DEFAULT_TOKEN2
from vocabulary import DEFAULT_TAG

hidden_size = 200
batch_size = 4
n_epoch = 10
lr = 1e-4 # learning rate
momentum = 0.9

class Decoder(nn.Module):
    def __init__(self, n_features, num_classes):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(n_features, num_classes)
        nn.init.normal_(self.fc1.weight.data)
        self.fc1.weight.data *= np.sqrt(2 / n_features)
        self.fc1.bias.data.fill_(0.0)

    def forward(self, x):
        return self.fc1(x)

def sequence_model(model, trees, samples, sents, pos_tags, vocab, tag_to_ind_map):
	# vectorized word_tag is the input to 1st layer Bi-LSTM cell
	n_features = len_of_vectorized_word_tag(trees[0], vocab, tag_to_ind_map) 

	print("num features per cell in 1st layer Bi-LSTM {}, num sentences {}".\
		format(n_features, len(sents)))

	print("Running LSTM sequence model")

	# first layer Bi-LSTM - input is sequence of words inside a sentence
	lstm1 = nn.LSTM(input_size=n_features, hidden_size=hidden_size, bidirectional=True)
	# second layer Bi-LSTM - input is sequence of EDUS inside a tree
	lstm2 = nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, bidirectional=True)

	# batch_size = min(batch_size, len(trees))

	num_classes = len(action_to_ind_map)
	decoder = Decoder(6 * hidden_size, num_classes)
	model._clf = decoder
	print(decoder)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(decoder.parameters(), lr=lr)
	print(optimizer)

	for epoch in range(1, n_epoch + 1):
		print("epoch {}".format(epoch))
		virt_mem = psutil.virtual_memory()
		print("memory available {} MB , mem used {} MB".format(\
			virt_mem[1] / 10 ** 6, virt_mem[3] / 10 ** 6))
		encoder_forward(lstm1, lstm2, trees, vocab, tag_to_ind_map)

		for x_vecs, y_labels in extract_hidden_edus_subtrees_next(trees, samples, vocab):
			optimizer.zero_grad() # zero the gradient buffers
			scores = decoder(Variable(x_vecs))
			loss = criterion(scores, Variable(y_labels))
			loss.backward()
			optimizer.step()
			
def extract_hidden_edus_subtrees_next(trees, samples, vocab):
	n_samples = len(samples)
	rand_samples = np.arange(n_samples)
	np.random.shuffle(rand_samples)
	x_vecs = []
	y_labels = []

	for i in range(1, n_samples + 1):
		sample_ind = rand_samples[i - 1]
		sample = samples[sample_ind]
		inp_vec = []
		for edu_i, edu_j in sample._spans:
			inp_vec += gen_edus_subtree_hidden_repr(sample._tree, edu_i, edu_j, vocab)
		inp_vec = torch.tensor(inp_vec)
		x_vecs.append(inp_vec)
		y_labels.append(action_to_ind_map[samples[sample_ind]._action])
		if i % batch_size == 0 or i == n_samples:
			y_labels = torch.tensor(y_labels, dtype=torch.long)
			# A two dimension tensor of dimension batch_size * (6 * hidden_size)
			x_vecs = torch.stack(x_vecs)
			yield x_vecs, y_labels
			x_vecs = []
			y_labels = []

def encoder_forward(lstm1, lstm2, trees, vocab, tag_to_ind_map):
	# Return a list of tensors where each represents a sequence of words 
	# inside a sentence, and has a shape of sent_len * vectorized concatenated word tag len
	# Tensors are sorted in reversal order by the number of their rows (sent_len)
	# Extract the inputs of the first Bi-LSTM layer 
	for x_vecs, batch_to_sent_ind, tree in prepare_sents_as_inp_vecs_next(trees, vocab, tag_to_ind_map):
		# Return one tensor of dimenion n_sents in tree * max_sent_len * hidden_size
		encoded_words, _ = lstm1(x_vecs)
		# Based on the encoded words, the outputs of the LSTM, the edus representations are calculated. 
		calc_edus_representations(tree, vocab, encoded_words, batch_to_sent_ind)

	# Preprare input vectors for the second Bi-LSTM encoder
	# Return a tensor of dimension batch_size * max edus seq len * (2*hidden_size)
	for x_vecs, batch_to_tree_ind in prepare_edus_seq_as_inp_vecs_next(trees, vocab, \
		tag_to_ind_map):
		encoded_edus, _ = lstm2(x_vecs)
		set_encoded_edus(trees, encoded_edus, batch_to_tree_ind)
		
def calc_edus_representations(tree, vocab, encoded_words, batch_to_sent_ind):
	"""
		For each edu in tree, calculate its representation by average pooling of
		the encoded words inside that edu. These representations is used as input vectors  
		for the EDUS sequence encoder.
	"""
	subset_size = len(batch_to_sent_ind)
	last_edu_ind = len(tree._EDUS_table) - 1

	if len(tree._edu_represent_table) == 1:
		tree._edu_represent_table += [0] * (len(tree._EDUS_table) - 1)

	for i in range(subset_size):
		sent_ind = batch_to_sent_ind[i]
		edu_ind = tree._sent_to_first_edu_ind[sent_ind]

		while edu_ind <= last_edu_ind and tree._edu_to_sent_ind[edu_ind] == sent_ind:
			# s, t are computed from 1
			s, t = tree._edus_seg_in_sent[edu_ind]
			edu_represent = 0
			for k in range(s, t + 1):
				edu_represent += encoded_words[i, k - 1, :] # dim 400
		
			tree._edu_represent_table[edu_ind] = edu_represent / (t - s + 1)
			edu_ind += 1

def set_encoded_edus(trees, encoded_edus, batch_to_tree_ind):
	for i in range(len(encoded_edus)):
		j = batch_to_tree_ind[i]
		tree = trees[j]
		if len(tree._encoded_edu_table) == 1:
			tree._encoded_edu_table += [0] * (len(tree._EDUS_table) - 1)
		for k in range(1, len(tree._EDUS_table[1:]) + 1):
			tree._encoded_edu_table[k] = encoded_edus[i, k - 1, :]

def prepare_edus_seq_as_inp_vecs_next(trees, vocab, tag_to_ind_map, batch_size=1):
	n_trees = len(trees)
	tuples = [] # tuples of tree_ind and seq_edus

	rand_indx = np.arange(n_trees)
	np.random.shuffle(rand_indx)

	for i in range(1, n_trees + 1):
		tree_ind = rand_indx[i - 1]
		tree = trees[tree_ind]
		edus_seq = tree._edu_represent_table[1:]
		# tensor dim is edus_seq_len * (2 * hidden_size)
		tuples.append((tree_ind, torch.stack(edus_seq)))
		if i % batch_size == 0 or i == n_trees:
			# sort by the length of edus seq in descendent order
			# x_vecs is a list of tensors of shape edus seq len * (2 * hidden_size)
			[batch_to_tree_ind, x_vecs] = sort_tuples_and_split(tuples, True)
			# x_padded_vecs is a tensor of n_sents * max_sent_len * n_features
			x_padded_vecs = pad_sequence(x_vecs, batch_first=True)
			yield x_padded_vecs, batch_to_tree_ind
			tuples = []

def prepare_sents_as_inp_vecs_next(trees, vocab, tag_to_ind_map):
	n_trees = len(trees)
	rand_indx = np.arange(n_trees)
	np.random.shuffle(rand_indx)
	tuples = [] # tuples of 'batch to sent ind' and 'vectorized word_tag seq'

	for i in range(1, n_trees + 1):
		tree = trees[rand_indx[i - 1]]
		sent_ind = 0
		last_sent_ind = len(tree._sent_tokenized_table) - 1
		for words, tags in list(zip(tree._sent_tokenized_table[1:], \
			tree._sent_pos_tags_table[1:])):
			sent_ind += 1
			vect_words_tags = []
			vect_words = gen_vectorized_words(words, vocab)
			vect_tags = gen_vectorized_tags(tags, tag_to_ind_map)
			for vect_word, vect_tag in list(zip(vect_words, vect_tags)):
				vect_words_tags.append(vect_word + vect_tag) # concatenation
			# tensor shape is the words_seq_len * vect_word_tag_len
			tuples.append((sent_ind, torch.tensor(vect_words_tags)))   
			if sent_ind % batch_size == 0 or sent_ind == last_sent_ind:
				# sort by the length of sents in descendent order
				# x_vecs is a list of tensors of shape sent len * word_tag_vect_len
				[batch_to_sent_ind, x_vecs] = sort_tuples_and_split(tuples, True)
				# stack a list of tensors along a new dimension (batch), and pad them to an equal length
				# x_padded_vecs is a tensor of n_sents * max_sent_len * n_features
				x_padded_vecs = pad_sequence(x_vecs, batch_first=True)
				yield x_padded_vecs, batch_to_sent_ind, tree
				tuples = []

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

def gen_edus_subtree_hidden_repr(tree, edu_i, edu_j, vocab):
	"""
		Generate subtree hidden representation by average pooling
		of its covering edus { edu_i, edu_i_1... edu_j}
	"""
	if edu_i == 0:
		return gen_def_edu_hidden_repr(vocab)
	subtree_repr = 0

	for k in range(edu_i, edu_j + 1):
		subtree_repr += tree._encoded_edu_table[k]

	return (subtree_repr / (edu_j - edu_i + 1)).tolist()