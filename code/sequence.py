import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import math

from features import extract_seq_features

def sequence_model(model, trees, samples, sents, pos_tags, vocab, tag_to_ind_map, \
	seq_len, n_epoch=10, batch_size=32):

	[seq] = extract_seq_features(trees, sents, pos_tags, vocab, tag_to_ind_map, 1)
	print(seq.shape)
	n_features = seq.shape[1] # a shape of sent_len * word vec dim

	hidden_size = 200
	batch_size = min(len(sents), batch_size)

	lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, bidirectional=True)

	n_subsets = math.ceil(len(sents) / batch_size)

	for epoch in range(1, n_epoch + 1):
		print("epoch {}".format(epoch))
		for i in range(n_subsets):
			x_vecs = extract_seq_features(trees, sents, pos_tags, vocab, tag_to_ind_map, batch_size)
			x_vecs = pad_sequence(x_vecs, batch_first=True, total_length=seq_len)
			outs, _ = lstm(x_vecs) # batch_size * seq_len * n_features
			



	