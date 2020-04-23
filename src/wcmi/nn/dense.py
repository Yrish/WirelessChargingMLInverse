# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
The dense network (the first model).

See `README.md` for more information.
"""

import torch.nn as nn

from wcmi.nn import modules

class Dense(modules.WCMIModule):
	"""
	The architecture for the dense model.
	"""
	def __init__(self, *args, **kwargs):
		"""
		Initialize a Dense module with extra parameters.

		Depending on parameters, optionally automatically load the parameters
		(weights and biases) from a file.
		"""

		# Parent initialization.
		super().__init__(*args, **kwargs)

		# Set attributes.
		#self.foo = foo

		# Set the neural network.
		self.net = nn.Sequential(
		)
