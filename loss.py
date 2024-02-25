import numpy as np
class MSELoss:
	def __init__(self):
		pass

	@staticmethod
	def forward(output, target):
		return np.square((output - target)**2).mean()
