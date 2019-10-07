import torch
import torch.nn as nn
from sequence_features import prepare_sents_as_inp_vecs_next
from sequence_features import prepare_edus_seq_as_inp_vecs
from general import print_memory_usage

def encoder_forward(lstm1, lstm2, samples, vocab, tag_to_ind_map, batch_size, use_def=False):
	# Return a list of tensors where each represents a sequence of words 
	# inside a sentence, and has a shape of sent_len * vectorized concatenated word tag len
	# Tensors are sorted in reversal order by the number of their rows (sent_len)
	# Extract the inputs of the first Bi-LSTM layer 
	for x_vecs, batch_to_sent_ind, batch_to_tree in prepare_sents_as_inp_vecs_next(\
		samples, vocab, tag_to_ind_map, batch_size, use_def):
		# print_memory_usage('4')
		# Return one tensor of dimenion n_sents in tree * max_sent_len * hidden_size
		encoded_words, _ = lstm1(x_vecs)
		# print_memory_usage('5')
		# Based on the encoded words, the outputs of the LSTM, the edus representations are calculated. 
		calc_edus_representations(vocab, encoded_words, batch_to_sent_ind, batch_to_tree)

	# print_memory_usage('1')
	# Preprare input vectors for the second Bi-LSTM encoder
	# Return a tensor of dimension batch_size * max edus seq len * (2*hidden_size)
	for x_vecs, batch_to_tree, batch_to_span in prepare_edus_seq_as_inp_vecs(\
		samples, vocab, tag_to_ind_map, batch_size):
		encoded_edus, _ = lstm2(x_vecs)
		set_encoded_edus(encoded_edus, batch_to_tree, batch_to_span)
	# print_memory_usage('2')

def calc_edus_representations(vocab, encoded_words, batch_to_sent_ind, batch_to_tree):
	"""
		For each edu in tree, calculate its representation by average pooling of
		the encoded words inside that edu. These representations is used as input vectors  
		for the EDUS sequence encoder.
	"""
	for i in range(len(batch_to_sent_ind)):
		sent_ind = batch_to_sent_ind[i]
		tree = batch_to_tree[i]
		last_edu_ind = len(tree._EDUS_table) - 1
		if len(tree._edu_represent_table) == 1:
			tree._edu_represent_table += [0] * (len(tree._EDUS_table) - 1)

		edu_ind = tree._sent_to_first_edu_ind[sent_ind]

		while edu_ind <= last_edu_ind and tree._edu_to_sent_ind[edu_ind] == sent_ind:
			# s, t are computed from 1
			s, t = tree._edus_seg_in_sent[edu_ind]
			edu_represent = 0
			for k in range(s, t + 1):
				edu_represent += encoded_words[i, k - 1, :] # dim 400
		
			tree._edu_represent_table[edu_ind] = edu_represent / (t - s + 1)
			edu_ind += 1

def set_encoded_edus(encoded_edus, batch_to_tree, batch_to_span):
	for i in range(len(encoded_edus)):
		tree = batch_to_tree[i]
		s, t = batch_to_span[i]
		if len(tree._encoded_edu_table) == 1:
			tree._encoded_edu_table += [0] * (len(tree._EDUS_table) - 1)
		for k in range(s, t + 1):
			tree._encoded_edu_table[k] = encoded_edus[i, k - s, :]