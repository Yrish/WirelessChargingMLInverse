# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
The GAN network (the second model).

See `README.md` for more information.
"""

import torch.nn as nn

from wcmi.nn import modules

# The default --gan-n value: the default number of additional input neurons to
# the generator beyond the 5 desired simulation output values.
default_gan_n = 8

class GAN(modules.WCMIModule):
	DEFAULT_GAN_N = default_gan_n

	"""
	The architecture for the dense model.
	"""
	def __init__(self, gan_n=None, *args, **kwargs):
		"""
		Initialize a Dense module with extra parameters.

		Depending on parameters, optionally automatically load the parameters
		(weights and biases) from a file.
		"""

		# Parent initialization.
		super().__init__(*args, **kwargs)

		# Set attributes.
		if gan_n is None:
			gan_n = default_gan_n

		# Set the neural network.
		self.net = nn.Sequential(
		)

	# Property: gan_n
	@property
	def gan_n(self):
		if self.checkpoint_extra['gan_n'] is None:
			self.checkpoint_extra['gan_n'] = default_gan_n
		return self.checkpoint_extra['gan_n']
	@gan_n.setter
	def gan_n(self, gan_n):
		if gan_n is None:
			gan_n = default_gan_n
		self.checkpoint_extra['gan_n'] = gan_n
