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

	# The default gan_n value.
	DEFAULT_GAN_N = default_gan_n

	# Label values for the discriminator.
	GENERATED_LABEL_ITEM = 0
	REAL_LABEL_ITEM = 1

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

		ADVERSARIAL
			Feed data through both the generator and the discriminator.
		"""
		DEFAULT            = 0
		GENERATOR_ONLY     = 1
		DISCRIMINATOR_ONLY = 2
		ADVERSARIAL        = 3

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

	# Utility: get input dimension: num_sim_inputs + gan_n
	@property
	def num_inputs_gen(self):
		"""Utility: get input dimension: num_sim_inputs + gan_n"""
		return self.simulation_info.num_sim_inputs + self.gan_n
	@property
	def num_outputs_gen(self):
		return self.simulation_info.num_sim_outputs

	def initialize_layers(self):
		"""
		Initialize the neural network architecture.
		"""

		# Set the neural network architecture.

		self.generator = nn.Sequential(
			nn.Bilinear(self.gan_n, self.num_sim_outputs, 90),
			nn.LeakyReLU(0.1),
			nn.Dropout(p=0.02),
			nn.Linear(90, self.num_sim_inputs),
			nn.Tanh() if data.is_standardized_negative() else nn.LeakyReLU(0.1),
			#nn.BatchNorm1d(self.simulation_info.num_sim_inputs),
		)

		self.discriminator = nn.Sequential(
			nn.Bilinear(self.simulation_info.num_sim_outputs, self.simulation_info.sum_sim_inputs, 90),
			nn.LeakyReLU(0.1),
			nn.Dropout(p=0.02),
			nn.Linear(90, 1),
			nn.Sigmoid(),
			#nn.BatchNorm1d(self.simulation_info.num_sim_inputs),
		)

	def get_subnetwork_selection(self, subnetwork_selection=GANSubnetworkSelection.DEFAULT):
		"""
		Reduce a GANSubnetworkSelection value to GENERATOR_ONLY,
		DISCRIMINATOR_ONLY, or ADVERSARIAL.
		"""
		if subnetwork_selection is None:
			subnetwork_selection = GANSubnetworkSelection.DEFAULT
		if subnetwork_selection == GANSubnetworkSelection.DEFAULT:
			subnetwork_selection = self.default_subnetwork_selection
		if subnetwork_selection is None or subnetwork_selection == GANSubnetworkSelection.DEFAULT:
			subnetwork_selection = DEFAULT_DEFAULT_SUBNETWORK_SELECTION
		if subnetwork_selection is None or subnetwork_selection == GANSubnetworkSelection.DEFAULT:
			subnetwork_selection = GANSubnetworkSelection.GENERATOR_ONLY
		return subnetwork_selection

	def set_default_subnetwork_selection(self, subnetwork_selection=DEFAULT_DEFAULT_SUBNETWORK_SELECTION):
		"""
		Set the subnetwork chosen when using the GAN unless explicitly
		overridden.

		Omit the `subnetwork_selection` parameter to restore the default,
		`GENERATOR_ONLY`.

		To keep the current value, pass DEFAULT.
		"""
		self.default_subnetwork_selection = self.get_subnetwork_selection(subnetwork_selection)

	def includes_generator(self, subnetwork_selection=GANSubnetworkSelection.DEFAULT):
		"""
		Determine whether subnetwork selection contains the generator.
		"""
		# Resolve the subnetwork selection.
		subnetwork_selection = self.get_subnetwork_selection(subnetwork_selection)

		# Determine whether subnetwork selection contains the generator.
		return subnetwork_selection in [
			GANSubnetworkSelection.GENERATOR_ONLY,
			GANSubnetworkSelection.ADVERSARIAL,
		]

	def includes_discriminator(self, subnetwork_selection=GANSubnetworkSelection.DEFAULT):
		"""
		Determine whether subnetwork selection contains the discriminator.
		"""

		# Resolve the subnetwork selection.
		subnetwork_selection = self.get_subnetwork_selection(subnetwork_selection)

		# Determine whether subnetwork selection contains the discriminator.
		return subnetwork_selection in [
			GANSubnetworkSelection.DISCRIMINATOR_ONLY,
			GANSubnetworkSelection.ADVERSARIAL,
		]

	def includes_both(self, subnetwork_selection=GANSubnetworkSelection.DEFAULT):
		"""
		Determine whether the subnetwork selection contains both the generator
		and the discriminator.
		"""

		# Resolve the subnetwork selection.
		subnetwork_selection = self.get_subnetwork_selection(subnetwork_selection)

		# Determine whether the subnetwork selection contains both the
		# generator and the discriminator.
		return self.includes_generator(subnetwork_selection) and self.includes_discriminator(subnetwork_selection)

	def forward(self, *input, subnetwork_selection=GANSubnetworkSelection.DEFAULT, **kwargs):
		"""
		Override `forward` to not destandardize output that leaves the
		discriminator and to not standardize GAN n input to the generator.

		(Non-GAN n generator input is still standardized, as is the generator's
		output.)

		The generator takes two inputs: the first is the 5 desired simulation
		inputs, just like the input to a Dense model; but the second is the GAN
		n generating values, which is not standardized by this model.

		The discriminator also takes two inputs: the first is the 5 desired
		simulation inputs, the same to the generator; and the second is 7
		simulation output values, which the discriminator must determine
		whether they're real or generated as its single output.  All input to
		the discriminator is standardized, but no output from the discriminator
		is.

		Only standardize non-GAN-n generator input, all generator output,
		and all discriminator input.
		"""
		# Resolve the subnetwork selection.
		subnetwork_selection = self.get_subnetwork_selection(subnetwork_selection)

		# Don't destandardize output that leaves the discriminator.
		standardize_input_only = self.includes_discriminator(subnetwork_selection)

		# Don't destandardize non-GAN-n generator input (2nd input), but
		# standardize all other input.
		if self.includes_generator(subnetwork_selection):
			standardize_input_mask = [True, False]
		else:
			standardize_input_mask = None  # i.e. [True, True]

		# Selectively standardize and forward.
		# (The construction lets our keywords themselves be overridden.
		return super().forward(
			*input,
			**{
				**dict(
					standardize_input_only=standardize_input_only,
					standardize_input_mask=standardize_input_mask,
				),
				**kwargs,
			},
		)

	def forward_with_standardized(self, *input, subnetwork_selection=GANSubnetworkSelection.DEFAULT, **kwargs):
		"""
		There are 3 possible forward passes:

		Generator only: 5 simulation output values and gan_n input values are fed to the network, and 7
		(num_sim_inputs) output values are returned.  This is the default.

		Discriminator only: 5 (num_sim_outputs) + 7 (num_sim_inputs) = 12
		inputs are input as a combination of simulation inputs and simulation
		outputs.  Output a single value (1 means real, 0 means generated).

		Adversarial (both): feed input through the generator and then the
		discriminator.

		By default, run the input through `self.net`.
		"""

		# Resolve the subnetwork selection.
		subnetwork_selection = self.get_subnetwork_selection(subnetwork_selection)

		# Route data through the appropriate subnetworks.
		if self.includes_generator(subnetwork_selection):
			sim_out, gan_gen = input
			sim_in = self.generator(sim_out, gan_gen)
		else:
			sim_out, sim_in = input

		if self.includes_discriminator(subnetwork_selection):
			output = self.discriminator(sim_out, sim_in)
		else:
			output = sim_in

		return output

# GENERATOR_ONLY.
default_default_subnetwork_selection = GAN.DEFAULT_DEFAULT_SUBNETWORK_SELECTION
