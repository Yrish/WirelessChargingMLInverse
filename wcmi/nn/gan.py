# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
The GAN network (the second model).

See `README.md` for more information.
"""

from enum import Enum

import torch.nn as nn

from wcmi.nn import modules
import wcmi.nn.data

# The default --gan-n value: the default number of additional input neurons to
# the generator beyond the 5 desired simulation output values.
default_gan_n = wcmi.nn.data.default_gan_n

class GAN(modules.WCMIModule):
	"""
	The architecture for the GAN model.
	"""

	DEFAULT_GAN_N = default_gan_n

	class GANSubnetworkSelection(Enum):
		"""
		A constant which specifies which GAN subnetwork to use.

		DEFAULT
			Use the default_subnetwork_selection value that the model
			current has.  The default value for *this* is GENERATOR_ONLY.

		GENERATOR_ONLY
			Use only the generator.

		DISCRIMINATOR_ONLY
			Use only the discriminator.

		TODO
			TODO
		"""
		DEFAULT            = 0
		GENERATOR_ONLY     = 1
		DISCRIMINATOR_ONLY = 2
		TODO               = 3

	DEFAULT_DEFAULT_SUBNETWORK_SELECTION = GANSubnetworkSelection.GENERATOR_ONLY

	def __init__(self, gan_n=None, default_subnetwork_selection=None, *args, **kwargs):
		"""
		Initialize a Gan module with extra parameters.

		Depending on parameters, optionally automatically load the parameters
		(weights and biases) from a file.
		"""

		# Parent initialization.
		super().__init__(*args, **kwargs)

		# Default arguments.
		if gan_n is None:
			gan_n = default_gan_n
		if default_subnetwork_selection is None:
			default_subnetwork_selection  = default_default_subnetwork_selection

		# Set persistent properties.
		#
		# Let the class's property handler store these.
		self.gan_n = gan_n

		# Set transient attributes.  These are not persistent.
		self.default_subnetwork_selection = default_subnetwork_selection

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

	def initialize_layers(self):
		# Set the neural network architecture.
		self.net = nn.Sequential(
			# TODO
		)

	def forward(self, x, subnetwork_selection=GANSubnetworkSelection.DEFAULT):
		"""
		There are 3 possible forward passes:

		Generator only: gan_n input values are fed to the network, and 7
		(num_sim_inputs) output values are returned.  This is the default.

		Discriminator only: 7 (num_sim_inputs) + 5 (num_sim_outputs) = 12
		inputs are input as a combination of simulation inputs and simulation
		outputs.  Output a binary value (1 for real, 0 for generated).

		By default, run the input through `self.net`.
		"""
		x = self.net(*input)
		return x

# GENERATOR_ONLY.
default_default_subnetwork_selection = GAN.DEFAULT_DEFAULT_SUBNETWORK_SELECTION
