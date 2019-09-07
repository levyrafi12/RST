import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import math

from features import extract_sents_features

def sequence_model(model, trees, samples, sents, pos_tags, vocab, tag_to_ind_map, \
	seq_len, n_epoch=10, batch_size=32):

	# Each example is represented by a single tensor/sequence
	x_vecs, _ = extract_sents_features([trees[0]._sent_tokenized_table[1]], \
		[trees[0]._sent_pos_tag_table[1]], vocab, tag_to_ind_map) 
	n_features = x_vecs[0].shape[1] # shape is sent_len * word_embed_len

	print("num features {}, num sentences {}, max sentence length {}".\
		format(n_features, len(sents), seq_len))
	print("Running LSTM sequence model")

	lstm = nn.LSTM(input_size=n_features, hidden_size=200, bidirectional=True)

	# n_batches = math.ceil(len(sents) / batch_size)

	for epoch in range(1, n_epoch + 1):
		print("epoch {}".format(epoch))
		for tree in trees:
			n_sents = len(tree._sent_tokenized_table[1:])
			# Return a list of tensors/sequences where each has a shape of sent_len * word_embed_len
			# tensors are sorted in reversal order by the number of their rows (sent_len)
			x_vecs, sent_ind_map = extract_sents_features(tree._sent_tokenized_table[1:], \
				tree._sent_pos_tag_table[1:], vocab, tag_to_ind_map)
			# stack a list of tensors along a new dimension (batch), and pad them to an equal length 
			x_padded_vecs = pad_sequence(x_vecs, batch_first=True)
			outs, _ = lstm(x_padded_vecs) # n_sents * max_sent_len * n_features
			# print("outs shape {}".format(outs.shape))
			for i in range(n_sents):
				sent_ind = sent_ind_map[i] # 'ind in batch' to sent ind
				# print("sent len {}".format(len(tree._sent_tokenized_table[sent_ind])))
				# print(tree._sent_tokenized_table[sent_ind])
				for j in range(len(outs[i])): # iterate over the words representations
					h_w = outs[i, j, :]
					# print("{} {}".format(j, h_w[0]))
			



	