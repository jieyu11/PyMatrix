import numpy as np
class Linear:
	def __init__(self, d_input, d_output):
		self.d_input = d_input
		self.d_output = d_output
		self._weights = 2 * (np.random.rand(d_input, d_output) - 0.5)
		self._bias = 2 * (np.random.rand(1, d_output) - 0.5)

	def forward(self, X):
		return X * self._weights + self._bias
	
	@property
	def weights(self):
		return self._weights
	
	@property
	def bias(self):
		return self._bias

class Relu:
	def __init__(self):
		pass

	def forward(self, X):
		return np.maximum(0., X)
