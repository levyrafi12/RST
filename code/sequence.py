import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import math
import numpy as np

from general import print_memory_usage
from rst_parser import evaluate
from sequence_features import extract_edus_subtrees_hidden_repr
from sequence_features import len_of_vectorized_word_tag
from sequence_features import get_samples_subset_next
from sequence_encoder import encoder_forward
from relations_inventory import action_to_ind_map

hidden_size = 200
batch_size = 8
n_epoch = 10
lr = 1e-4 # learning rate
momentum = 0.9
print_every = 1

class Decoder(nn.Module):
    def __init__(self, n_features, num_classes):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(n_features, num_classes)
        nn.init.normal_(self.fc1.weight.data)
        self.fc1.weight.data *= np.sqrt(2 / n_features)
        self.fc1.bias.data.fill_(0.0)

    def forward(self, x):
        return self.fc1(x)

def sequence_model(model, samples, vocab, tag_to_ind_map, gen_dep):
	# vectorized word_tag is the input to 1st layer Bi-LSTM cell
	input_size = len_of_vectorized_word_tag(samples[0]._tree, vocab, tag_to_ind_map) 

	print("cell input size in 1st layer Bi-LSTM {}".format(input_size))

	print("Running LSTM sequence model")

	# first layer Bi-LSTM - input is sequence of words inside a sentence
	lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True)
	# second layer Bi-LSTM - input is sequence of EDUS inside a tree
	lstm2 = nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, bidirectional=True)

	num_classes = len(action_to_ind_map)
	decoder = Decoder(6 * hidden_size, num_classes)
	model._clf = decoder
	model._lstm1 = lstm1
	model._lstm2 = lstm2
	model._bs = 1
	print(decoder)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(decoder.parameters(), lr=lr)
	print(optimizer)

	print_memory_usage()
	n_match = 0

	for epoch in range(1, n_epoch + 1):
		i = 1
		for samples_subset in get_samples_subset_next(samples, batch_size):
			# print("encoder forward begin {}".format(i))
			encoder_forward(lstm1, lstm2, samples_subset, vocab, tag_to_ind_map, 1)
			# print("encoder forward done {}".format(i))
			i += 1
			x_vecs, y_labels = extract_edus_subtrees_hidden_repr(samples_subset, vocab)
			optimizer.zero_grad() # zero the gradient buffers
			scores = decoder(Variable(x_vecs)) 
			# indices[i] is the predicted action ind of sample i 
			indices = scores.max(1)[1] # dim=1 refers to rows, size(indices)=batch size
			loss = criterion(scores, Variable(y_labels))
			loss.backward()
			optimizer.step()
			if i % 100 == 0:
				print(i)
				print_memory_usage('3')

			n_match += np.sum([indices[j] == y_labels[j] for j in range(len(indices))])

		print("epoch {0} num matches = {1:.3f}% loss {2:.3f}".\
			format(epoch, n_match / len(samples) * 100, loss.item()))
		print_memory_usage()
		evaluate(model, vocab, tag_to_ind_map, gen_dep)
		gen_dep = False
		n_match = 0

