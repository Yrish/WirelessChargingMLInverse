# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
Module providing base classes for our models (neural networks).
"""

import torch
import torch.nn as nn
import sys

from wcmi import simulation

import wcmi.version

class WCMIModule(nn.Module):
	"""
	Base class with save and load utility methods.

	If simulation_info is None, it will be initialized to a default value.
	"""
	def __init__(
		self, load_model_path=None, save_model_path=None,
		auto_load_model=True, auto_initialize=True,
		simulation_info=None,
		*args, **kwargs
	):
		"""
		Initialize a WCMI module with extra parameters.

		Depending on parameters, optionally automatically load the parameters
		(weights and biases) from a file.
		"""

		# Parent initialization.
		super().__init__(*args, **kwargs)

		# Set properties.
		self.load_model_path = load_model_path
		self.save_model_path = save_model_path
		self.auto_load_model = auto_load_model
		self.auto_initialize = auto_initialize

		# Set persistent properties.
		self.simulation_info = simulation_info
		if self.simulation_info is None:
			self.simulation_info = simulation.simulation_info

		# Set other attributes.
		self.checkpoint_extra = {}

		# We haven't initialized the model yet.
		self.initialized = False

		# If auto_load_model is true and we have a path, initialize the model.
		if self.auto_load_model:
			if self.load_model_path is not None:
				self.load()

		# If auto_initialize and we haven't initialized the model yet, then
		# randomly initialize it.
		if self.auto_initialize:
			if not self.initialized:
				self.initialize_parameters()

	def save(self, save_model_path=None, update_save_model_path=True):
		"""
		Save the pytorch model.

		If the model was not initialized with a save path, it must be provided,
		or a WCMIError exception will be raised.
		"""

		# Make sure we have somewhere to save to.
		if save_model_path is None:
			save_model_path = self.save_model_path
		if save_model_path is None:
			raise WCMIError("error: WCMIModule.save() was called with no model path.  Where would it save to?")
		if update_save_model_path:
			self.save_model_path = save_model_path

		# Save the model.
		checkpoint = {
			"wcmi_version": wcmi.version.version,
			"state_dict": self.state_dict(),
			"simulation_info": self.simulation_info,
			"checkpoint_extra": self.checkpoint_extra,
		}
		return torch.save(checkpoint, save_model_path)

	def load(self, load_model_path=None, update_load_model_path=True, error_version=True, warn_version=True):
		"""
		Load a pytorch model.

		If the model was not initialized with a load path, it must be provided,
		or a WCMIError exception will be raised.
		"""

		# Make sure we have somewhere to load from.
		if load_model_path is None:
			load_model_path = self.load_model_path
		if load_model_path is None:
			raise WCMIError("error: WCMIModule.load() was called with no model path.  Where would it load from?")
		if update_load_model_path:
			self.load_model_path = load_model_path

		# Load the model.
		checkpoint = torch.load(load_model_path)
		self.wcmi_version = checkpoint["wcmi_version"]
		result = self.load_state_dict(checkpoint["state_dict"])
		self.simulation_info = checkpoint["simulation_info"]
		self.checkpoint_extra = checkpoint["checkpoint_extra"]
		self.initialized = True

		# Verify the version.
		model_version = self.wcmi_version
		our_version   = wcmi.version.version
		if not wcmi.version.version_compatible(model_version, our_version):
			msg = "error: WCMIModule.load() loaded a module with version {0:s} different from the current version {1:s}.".format(
				wcmi.version.version_str(model_version),
				wcmi.version.version_str(our_version),
			)
			if error_version:
				raise WCMIError("error: {0:s}".format(msg))
			if warning_version:
				print("warning: {0:s}".format(msg), file=sys.stderr)

		return result

	def initialize_parameters(self):
		"""
		Initialize the parameters.  Generally this is used when there is no
		model to be loaded.

		By default, no additional initialization is performed.
		"""
		pass

	def forward(self, *input):
		"""
		By default, run the input through `self.net`.
		"""
		x = self.net(*input)
		return x
