import torch
import torch.nn as nn

def sequence_model(model, trees, samples, vocab, tag_to_ind_map, seq_len):
	hidden_size = 200
	batch_size = 32

	lstm = nn.LSTM(, hidden_size=hidden_size, bidirectional=True)


	