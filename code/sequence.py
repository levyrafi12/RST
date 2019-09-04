import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import math

from features import extract_seq_features

def sequence_model(model, trees, samples, sents, pos_tags, vocab, tag_to_ind_map, \
	seq_len, n_epoch=10, batch_size=32):

	# Each example is represented by a single tensor/sequence
	x_vecs = extract_seq_features(trees, sents, pos_tags, vocab, tag_to_ind_map, 1)
	n_features = x_vecs[0].shape[1] # a shape of sent_len * word_embed_len

	hidden_size = 200
	batch_size = min(len(sents), batch_size)

	print("num features {}, num sentences {}, batch size {}, max sentence length {}".\
		format(n_features, len(sents), batch_size, seq_len))
	print("Running LSTM sequence model")

	lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, bidirectional=True)

	n_batches = math.ceil(len(sents) / batch_size)

	for epoch in range(1, n_epoch + 1):
		print("epoch {}".format(epoch))
		for i in range(n_batches):
			# Return a list of tensors/sequences each of a shape: sent_len * word_embed_len
			x_vecs = extract_seq_features(trees, sents, pos_tags, vocab, tag_to_ind_map, batch_size)
			# stack a list of tensors along a new dimension, and pad them to an equal length 
			x_vecs = pad_sequence(x_vecs, batch_first=True)
			print(x_vecs.shape)
			outs, _ = lstm(x_vecs) # batch_size * seq_len * n_features
			print("outs shape {}".format(outs.shape))
			



	