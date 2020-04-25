# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
Module providing base classes for our models (neural networks).
"""

import numpy as np
import torch
import torch.nn as nn
import sys

from wcmi import simulation

import wcmi.nn.data as data
import wcmi.version

class WCMIModule(nn.Module):
	"""
	Base class with save and load utility methods.

	If simulation_info is None, it will be initialized to a default value.
	"""
	def __init__(
		self, load_model_path=None, save_model_path=None,
		auto_load_model=True, auto_initialize=True,
		simulation_info=None, standardize=data.standardize,
		normalize_population=data.normalize_population,
		normalize_bounds=data.normalize_bounds,
		normalize_negative=data.normalize_negative,
		population_mean_in=None, population_std_in=None,
		population_min_in=None, population_max_in=None,
		population_mean_out=None, population_std_out=None,
		population_min_out=None, population_max_out=None,
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
		self.standardize = standardize
		self.normalize_population = normalize_population
		self.normalize_bounds = normalize_bounds
		self.normalize_negative = normalize_negative
		self.population_mean_in = population_mean_in
		self.population_std_in = population_std_in
		self.population_min_in = population_min_in
		self.population_max_in = population_max_in
		self.population_mean_out = population_mean_out
		self.population_std_out = population_std_out
		self.population_min_out = population_min_out
		self.population_max_out = population_max_out

		# Set other persistent properties.
		self.checkpoint_extra = {}

		# Set other attributes.
		self.bounds_min_in  = torch.tensor(self.simulation_info.sim_output_mins)
		self.bounds_max_in  = torch.tensor(self.simulation_info.sim_output_maxs)
		self.bounds_min_out = torch.tensor(self.simulation_info.sim_input_mins)
		self.bounds_max_out = torch.tensor(self.simulation_info.sim_input_maxs)

		# We haven't initialized the model yet.
		self.initialized = False

		# Call the layer setup method.
		self.initialize_layers()

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
			"standardize": self.standardize,
			"normalize_population": self.normalize_population,
			"normalize_bounds": self.normalize_bounds,
			"normalize_negative": self.normalize_negative,
			"population_mean_in": self.population_mean_in,
			"population_std_in": self.population_std_in,
			"population_min_in": self.population_min_in,
			"population_max_in": self.population_max_in,
			"population_mean_out": self.population_mean_out,
			"population_std_out": self.population_std_out,
			"population_min_out": self.population_min_out,
			"population_max_out": self.population_max_out,
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
		self.standardize = checkpoint["standardize"]
		self.normalize_population = checkpoint["normalize_population"]
		self.normalize_bounds = checkpoint["normalize_bounds"]
		self.normalize_negative = checkpoint["normalize_negative"]
		self.population_mean_in = checkpoint["population_mean_in"]
		self.population_std_in = checkpoint["population_std_in"]
		self.population_min_in = checkpoint["population_min_in"]
		self.population_max_in = checkpoint["population_max_in"]
		self.population_mean_out = checkpoint["population_mean_out"]
		self.population_std_out = checkpoint["population_std_out"]
		self.population_min_out = checkpoint["population_min_out"]
		self.population_max_out = checkpoint["population_max_out"]
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

	def initialize_layers(self):
		"""
		Initialize the neural network's layers.

		Subclasses should define this method.
		"""
		raise NotImplementedError

	def initialize_parameters(self):
		"""
		Initialize the parameters.  Generally this is used when there is no
		model to be loaded.

		By default, no additional initialization is performed.
		"""
		if False:
			# Fix pytorch model returning NaNs.
			# c.f.  https://discuss.pytorch.org/t/manually-initialize-parameters/14337/2
			for p in self.parameters():
				p.data.fill_(0)
				if True:
					torch.rand(p.data.shape, out=p.data)

	def with_standardized(self, *input, forward=None, in_place=False):
		"""
		If standardization or normalization is enabled, apply it; else, leave
		the input untouched.
		"""
		# Argument defaults.
		if forward is None:
			forward = self.forward_with_standardized

		if in_place:
			input_standardized = input
		else:
			input_standardized = (*(x.clone() for x in input),)

		# Determine whether to standardize, normalize, or neither.
		if self.standardize:
			xs = [(x - self.population_mean_in)/self.population_std_in for x in input]
			xs = forward(*xs)
			x = self.population_std_out * xs + self.population_mean_out
			return x
		elif self.normalize_population:
			data_min_in    = self.population_min_in
			data_max_in    = self.population_max_in
			data_range_in  = data_max_in - data_min_in
			data_min_out   = self.population_min_out
			data_max_out   = self.population_max_out
			data_range_out = data_max_out - data_min_out
			if not self.normalize_negative:
				xs = [(x - data_min_in)/data_range_in for x in input]
				xs = forward(*xs)
				x = data_range_out * xs + data_min_out
				return x
			else:
				xs = [2*(x - data_min_in)/data_range_in - 1 for x in input]
				xs = forward(*xs)
				x = data_range_out * ((xs+1)/2) + data_min_out
				return x
		elif self.normalize_bounds:
			data_min_in    = self.bounds_min_in
			data_max_in    = self.bounds_max_in
			data_range_in  = data_max_in - data_min_in
			data_min_out   = self.bounds_min_out
			data_max_out   = self.bounds_max_out
			data_range_out = data_max_out - data_min_out
			if not self.normalize_negative:
				xs = [(x - data_min_in)/data_range_in for x in input]
				xs = forward(*xs)
				x = data_range_out * xs + data_min_out
				return x
			else:
				xs = [2*(x - data_min_in)/data_range_in - 1 for x in input]
				xs = forward(*xs)
				x = data_range_out * ((xs+1)/2) + data_min_out
				return x
		else:
			x = self.net(*input)
			return x

	def forward(self, *input):
		"""
		By default, run the input through `self.net`.
		"""
		x = self.with_standardized(*input, forward=self.forward_with_standardized, in_place=False)
		return x

	def forward_with_standardized(self, *input):
		"""
		By default, run the input through `self.net`.
		"""
		x = self.net(*input)
		return x
