import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import math

from features import extract_sents_features

def sequence_model(model, trees, samples, sents, pos_tags, vocab, tag_to_ind_map, \
	n_epoch=10, batch_size=32):

	# Each example is represented by a single tensor/sequence
	x_vecs, _ = extract_sents_features([trees[0]._sent_tokenized_table[1]], \
		[trees[0]._sent_pos_tags_table[1]], vocab, tag_to_ind_map) 
	n_features = x_vecs[0].shape[1] # shape is sent_len * word_embed_len

	print("num features {}, num sentences {}".format(n_features, len(sents)))
	print("Running LSTM sequence model")

	lstm = nn.LSTM(input_size=n_features, hidden_size=200, bidirectional=True)

	# n_batches = math.ceil(len(sents) / batch_size)

	for epoch in range(1, n_epoch + 1):
		print("epoch {}".format(epoch))
		for tree in trees:
			# print(tree._fname)
			# Return a list of tensors/sequences where each has a shape of sent_len * word_embed_len
			# tensors are sorted in reversal order by the number of their rows (sent_len)
			# first bi-lstm layer
			x_vecs, sent_ind_map = extract_sents_features(tree._sent_tokenized_table[1:], \
				tree._sent_pos_tags_table[1:], vocab, tag_to_ind_map)
			# stack a list of tensors along a new dimension (batch), and pad them to an equal length 
			x_padded_vecs = pad_sequence(x_vecs, batch_first=True)
			outs, _ = lstm(x_padded_vecs) # n_sents * max_sent_len * n_features
			calc_edus_embeddings(tree, sent_ind_map, outs)
			
			
def calc_edus_embeddings(tree, sent_ind_map, outs):
	# print(tree._fname)
	# print(outs.shape)
	n_sents = len(sent_ind_map)
	sent_to_batch_ind = [0] * (n_sents + 1)
	for batch_ind in range(n_sents):
		sent_ind = sent_ind_map[batch_ind]
		sent_to_batch_ind[sent_ind] = batch_ind

	edu_ind = 1
	n_edus = len(tree._edu_to_sent_ind[1:])
	for sent_ind in range(1, n_sents + 1):
		batch_ind = sent_to_batch_ind[sent_ind]
		edu_embed = 0

		while edu_ind <= n_edus and tree._edu_to_sent_ind[edu_ind] == sent_ind:
			# s, t are computed from 1
			s, t = tree._edus_seg_in_sent[edu_ind]
			# print("{} {} {} {}".format(tree._fname, s, t, outs.shape[1]))
			edu_emb = 0
			assert t <= outs.shape[1], \
				print("{} {} sent {}".format(s, t, tree._sent_tokenized_table[sent_ind]))
			for k in range(s, t + 1):
				edu_emb += outs[batch_ind, k - 1, :] # dim 400
		
			tree._edu_embed_table.append(edu_emb / (t - s + 1))
			edu_ind += 1

		assert t == len(tree._sent_tokenized_table[sent_ind]), \
			print("{} sent ind {} edu ind {} edu {} sent {}".format(tree._fname, sent_ind, \
				edu_ind - 1, tree._edu_tokenized_table[edu_ind - 1], \
				tree._sent_tokenized_table[sent_ind]))