import numpy as np

class Model(object):
	def __init__(self, name):
		self._name = name
		self._proj_mat = np.array([])
		self._clf = None

	def is_proj_mat(self):
		return len(self._proj_mat) > 0

