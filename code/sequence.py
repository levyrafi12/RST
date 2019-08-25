import torch
import torch.nn as nn
from nn.utils.rnn import pad_sequence

from features extract_seq_features

def sequence_model(model, trees, samples, sents, pos_tags, vocab, tag_to_ind_map, \
	seq_len, n_epoch=10, batch_size=32):

	[seq] = extract_seq_features(trees, sents, pos_tags, vocab, tag_to_ind_map, 1) 
	n_features = seq.shape[1] # a shape of sent_len * word vec dim

	hidden_size = 200
	batch_size = 32

	lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, bidirectional=True)

	n_subsets = math.ceil(len(samples) / batch_size)

	for epoch in range(for i in range(n_subsets)1, n_epoch + 1):
		for i in range(n_subsets):
			x_vecs = extract_seq_features(trees, sents, pos_tags, vocab, tag_to_ind_map, batch_size)
			# inputs = pad_sequence(inputs)
			outs, _ = lstm(x_vecs)
			



	