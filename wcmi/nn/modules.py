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
from wcmi.log import logger

import wcmi.nn.data as data
import wcmi.version

class WCMIModule(nn.Module):
	"""
	Base class with save and load utility methods.

	If simulation_info is None, it will be initialized to a default value.

	If reverse is set, then instead of predicting simulation inputs for desired
	simulation outputs, the neural network learns to predict simulation outputs
	from simulation inputs.
	"""
	def __init__(
		self, load_model_path=None, save_model_path=None,
		auto_load_model=True, auto_initialize=True, simulation_info=None,
		reverse=False, standardize=data.standardize,
		normalize_population=data.normalize_population,
		normalize_bounds=data.normalize_bounds,
		normalize_negative=data.normalize_negative, population_mean_in=None,
		population_std_in=None, population_min_in=None, population_max_in=None,
		population_mean_out=None, population_std_out=None,
		population_min_out=None, population_max_out=None,
		standardize_bounds_multiple=False,
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
		self.reverse = reverse
		if self.reverse is None:
			self.reverse = simulation.reverse
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
		# standardize_bounds_multiple: Are there multiple sets of
		# standardization parameters to be applied element-wise to the inputs,
		# rather than one set to apply to all inputs?
		self.standardize_bounds_multiple = standardize_bounds_multiple

		# Set other persistent properties.
		if not hasattr(self, "checkpoint_extra"):
			self.checkpoint_extra = {}

		# Set other attributes.
		self.bounds_min_in  = torch.tensor(self.simulation_info.sim_output_mins)
		self.bounds_max_in  = torch.tensor(self.simulation_info.sim_output_maxs)
		self.bounds_min_out = torch.tensor(self.simulation_info.sim_input_mins)
		self.bounds_max_out = torch.tensor(self.simulation_info.sim_input_maxs)

		# We haven't initialized the model yet.
		self.initialized = False

		# If auto_load_model is true and we have a path, initialize the model.
		need_load = False
		if self.auto_load_model:
			if self.load_model_path is not None:
				need_load = True
		if need_load:
			self.load(skip_state_dict=True)

		# Call the layer setup method.
		self.initialize_layers()

		# Now we have both reverse, etc. and the layers.  Load the state dict.
		if need_load:
			self.load(skip_non_state_dict=True)

		# If auto_initialize and we haven't initialized the model yet, then
		# randomly initialize it.
		if self.auto_initialize:
			if not self.initialized:
				self.initialize_parameters()

	def save(self, save_model_path=None, update_save_model_path=True, logger=logger):
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
			"reverse": self.reverse,
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
			"standardize_bounds_multiple": self.standardize_bounds_multiple,
			"checkpoint_extra": self.checkpoint_extra,
		}
		return torch.save(checkpoint, save_model_path)

	def load(self, load_model_path=None, update_load_model_path=True, error_version=False, warn_version=True, skip_state_dict=False, skip_non_state_dict=False, logger=logger):
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
		if not skip_state_dict:
			result = self.load_state_dict(checkpoint["state_dict"])
		else:
			result = None
		if not skip_non_state_dict:
			self.simulation_info = checkpoint["simulation_info"]
			self.checkpoint_extra = checkpoint["checkpoint_extra"]
			self.standardize = checkpoint["standardize"]
			self.reverse = checkpoint["reverse"]
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
			self.standardize_bounds_multiple = checkpoint["standardize_bounds_multiple"]
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
				logger.warning("warning: {0:s}".format(msg), file=sys.stderr)

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

	@staticmethod
	def standardize_val(
		v, skip_all_standardize=False, is_standardize=False,
		is_normalize=False, is_normalize_negative=True,
		mean_in=None, std_in=None, min_in=None, max_in=None,
		mean_out=None, std_out=None, min_out=None, max_out=None,
		destandardize=False,
	):
		"""Standardize a single input according to the parameters."""
		if skip_all_standardize:
			return v
		elif is_standardize:
			if not destandardize:
				return (v - mean_in)/std_in
			else:
				return std_out * v + mean_out
		elif is_normalize:
			if not is_normalize_negative:
				if not destandardize:
					return (v - min_in)/(max_in - min_in)
				else:
					return (max_out - min_out) * v + min_out
			else:
				if not destandardize:
					return 2*(v - min_in)/(max_in - min_in) - 1
				else:
					return (max_out - min_out) * ((v+1)/2) + min_out
		else:
			return v

	@staticmethod
	def destandardize_val(*args, **kwargs):
		"""Destandardize a single output."""
		return WCMIModule.standardize_val(*args, **{**dict(destandardize=True), **kwargs})

	def with_standardized(self, *input, forward=None, standardize_input_only=False, standardize_input_mask=None, in_place=False, standardize_bounds_multiple=None, **kwargs):
		"""
		If standardization or normalization is enabled, apply it; else, leave
		the input untouched.
		"""
		# Argument defaults.
		if forward is None:
			forward = self.forward_with_standardized
		if standardize_bounds_multiple is None:
			standardize_bounds_multiple = self.standardize_bounds_multiple

		if in_place:
			input_standardized = input
		else:
			input_standardized = (*(x.clone() for x in input),)

		if not standardize_bounds_multiple:
			xs = [
				self.standardize_val(
					x,
					skip_all_standardize=standardize_input_mask is not None and not standardize_input_mask[i],
					is_standardize=self.standardize,
					is_normalize=self.normalize_population or self.normalize_bounds,

					mean_in=self.population_mean_in, std_in=self.population_std_in,
					mean_out=self.population_mean_out, std_out=self.population_std_out,

					min_in=self.population_min_in if self.normalize_population else self.bounds_min_in,
					max_in=self.population_max_in if self.normalize_population else self.bounds_max_in,
					min_out=self.population_min_out if self.normalize_population else self.bounds_min_out,
					max_out=self.population_max_out if self.normalize_population else self.bounds_max_out,
				)
				for i, x in enumerate(input)
			]
		else:
			xs = [
				self.standardize_val(
					x,
					skip_all_standardize=standardize_input_mask is not None and not standardize_input_mask[i],
					is_standardize=self.standardize,
					is_normalize=self.normalize_population or self.normalize_bounds,

					mean_in=self.population_mean_in[i], std_in=self.population_std_in[i],
					mean_out=self.population_mean_out[i], std_out=self.population_std_out[i],

					min_in=self.population_min_in[i] if self.normalize_population else self.bounds_min_in[i],
					max_in=self.population_max_in[i] if self.normalize_population else self.bounds_max_in[i],
					min_out=self.population_min_out[i] if self.normalize_population else self.bounds_min_out[i],
					max_out=self.population_max_out[i] if self.normalize_population else self.bounds_max_out[i],
				)
				for i, x in enumerate(input)
			]
		xs = forward(*xs, **kwargs)
		return self.destandardize_val(
			xs,
			skip_all_standardize=standardize_input_only,
			is_standardize=self.standardize,
			is_normalize=self.normalize_population or self.normalize_bounds,

			mean_in=self.population_mean_in, std_in=self.population_std_in,
			mean_out=self.population_mean_out, std_out=self.population_std_out,

			min_in=self.population_min_in if self.normalize_population else self.bounds_min_in,
			max_in=self.population_max_in if self.normalize_population else self.bounds_max_in,
			min_out=self.population_min_out if self.normalize_population else self.bounds_min_out,
			max_out=self.population_max_out if self.normalize_population else self.bounds_max_out,
		)

	def forward(self, *input, **kwargs):
		"""
		By default, run the input through `self.net` after handling
		standardization settings.
		"""
		x = self.with_standardized(*input, **kwargs)
		return x

	def forward_with_standardized(self, *input, **kwargs):
		"""
		By default, run the input through `self.net`.
		"""
		x = self.net(*input)
		return x

	def get_model_input_size(self, *args, **kwargs):
		"""
		By default, this is simulation_info.num_sim_outputs.

		Depending on the network, this might return multiple values.
		"""
		if not self.reverse:
			# desired sim output to predicted sim input.
			model_input_size  = self.simulation_info.num_sim_outputs
			model_output_size = self.simulation_info.num_sim_inputs
		else:
			# Straightforward neural network approximation for the ANSYS
			# simulation.
			model_input_size  = self.simulation_info.num_sim_inputs
			model_output_size = self.simulation_info.num_sim_outputs

		return model_input_size

	def get_model_output_size(self, *args, **kwargs):
		"""
		By default, this is simulation_info.num_sim_inputs.
		"""
		if not self.reverse:
			# desired sim output to predicted sim input.
			model_input_size  = self.simulation_info.num_sim_outputs
			model_output_size = self.simulation_info.num_sim_inputs
		else:
			# Straightforward neural network approximation for the ANSYS
			# simulation.
			model_input_size  = self.simulation_info.num_sim_inputs
			model_output_size = self.simulation_info.num_sim_outputs

		return model_output_size
