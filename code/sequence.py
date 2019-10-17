import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

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
encoder_bs = 8
decoder_bs = 1
n_epoch = 10
lr = 1e-4 # learning rate
grad_clip = 10 # gradient clipping
dropout_prob = 0.5
l2_regul = 1e-6 # l2 regularization factor

class Decoder(nn.Module):
    def __init__(self, n_features, num_classes):
        super(Decoder, self).__init__()
        # self.dropout = nn.Dropout(p=dropout_prob)
        self.fc1 = nn.Linear(n_features, num_classes)
        np.random.seed(1)
        nn.init.normal_(self.fc1.weight.data)
        self.fc1.weight.data *= np.sqrt(2 / n_features)
        self.fc1.bias.data.fill_(0.0)

    def forward(self, x):
    	# x = self.dropout(x)
        return self.fc1(x)

def sequence_model(model, samples, vocab, tag_to_ind_map, gen_dep):
	print("Running LSTM sequence model: num samples {}".format(len(samples)))
	# vectorized word_tag is the input to 1st layer Bi-LSTM cell
	input_size = len_of_vectorized_word_tag(samples[0]._tree, vocab, tag_to_ind_map) 

	print("cell input size of 1st encoder {}".format(input_size))

	# first layer Bi-LSTM - input is sequence of words inside a sentence
	lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, \
		batch_first=True)
	# second layer Bi-LSTM - input is sequence of EDUS inside a tree
	lstm2 = nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, bidirectional=True, \
		batch_first=True)

	lstm1_opt = optim.Adam(lstm1.parameters(), lr=lr, weight_decay=l2_regul)
	lstm2_opt = optim.Adam(lstm2.parameters(), lr=lr, weight_decay=l2_regul)

	num_classes = len(action_to_ind_map)
	decoder = Decoder(6 * hidden_size, num_classes)
	model._clf = decoder
	model._lstm1 = lstm1
	model._lstm2 = lstm2
	model._bs = decoder_bs
	print(decoder)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(decoder.parameters(), lr=lr, weight_decay=l2_regul)
	print(optimizer)

	print_memory_usage()
	n_match = 0

	for epoch in range(1, n_epoch + 1):
		i = 1
		for samples_subset in get_samples_subset_next(samples, decoder_bs):
			encoder_forward(lstm1, lstm2, samples_subset, vocab, tag_to_ind_map, encoder_bs)
			x_vecs, y_labels = extract_edus_subtrees_hidden_repr(samples_subset, vocab)
			optimizer.zero_grad() # zero the gradient buffers
			lstm1_opt.zero_grad()
			lstm2_opt.zero_grad()
			scores = decoder(x_vecs) 
			# indices[i] is the predicted action ind of sample i 
			indices = scores.max(1)[1] # dim=1 refers to rows, size(indices)=batch size
			loss = criterion(scores, y_labels)
			# check_grad(samples_subset)
			loss.backward()
			optimizer.step()
			lstm1_opt.step()
			lstm2_opt.step()
			clip_grad_norm_(decoder.parameters(), grad_clip)
			clip_grad_norm_(lstm1.parameters(), grad_clip)
			clip_grad_norm_(lstm2.parameters(), grad_clip)
			if i % 50 == 0:
				print(i)
				# print("dec inp {} {}".format(x_vecs.is_leaf, x_vecs.grad_fn))
				# print(torch.sum(decoder.fc1.weight))
				# enc1_w = getattr(lstm1, "weight_hh_l0")
				# print("enc1 weight {} {}".format(enc1_w.is_leaf, enc1_w.grad))
				# enc2_w = getattr(lstm2, "weight_hh_l0")
				# print("enc2 weight {} {}".format(enc2_w.is_leaf, enc2_w.grad))
				# print(torch.sum(getattr(lstm1, "weight_hh_l0")))
				# print(torch.sum(getattr(lstm2, "weight_hh_l0")))
				print_memory_usage()

			n_match += np.sum([indices[j] == y_labels[j] for j in range(len(indices))])
			i += 1

		print("epoch {0} num matches = {1:.3f}% loss {2:.3f}".\
			format(epoch, n_match / len(samples) * 100, loss.item()))
		print_memory_usage()
		evaluate(model, vocab, tag_to_ind_map, gen_dep)
		gen_dep = False
		n_match = 0

def check_grad(samples):
	for sample in samples:
		for s, t in sample._spans:
			for k in range(s, t + 1):
				if not sample._tree._edu_represent_table[k].grad:
					print("hello")
		for s, t in sample._sents_spans:
			for k in range(s, t + 1):
				if not sample._tree._encoded_edu_table[k].grad:
					print("hello2")
