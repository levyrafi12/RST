import numpy as np
from features import add_features_per_sample
from sequence_features import extract_edus_subtrees_hidden_repr_per_sample
from sequence_encoder import encoder_forward
from features import is_bag_of_words
from features import is_basic_feat
from features import get_word_encoding

class Model(object):
	def __init__(self, name):
		self._name = name
		self._proj_mat = np.array([])
		self._clf = None

	def is_proj_mat(self):
		return len(self._proj_mat) > 0

	def extract_input_vec(self, sample, vocab, tag_to_ind_map):
		if self._name == 'seq':
			encoder_forward(self._lstm1, self._lstm2, [sample], vocab, tag_to_ind_map, self._bs, True)
			# print(len(sample._tree._encoded_edu_table))
			# print("in eval sample spans {}".format(sample._spans))
			return extract_edus_subtrees_hidden_repr_per_sample(sample, vocab)

		_, x_vecs = add_features_per_sample(sample, vocab, tag_to_ind_map, True, 
			is_bag_of_words(self._name), is_basic_feat(self._name), \
			get_word_encoding(self._name))
		return x_vecs
		
