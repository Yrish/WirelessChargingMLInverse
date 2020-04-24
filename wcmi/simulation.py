# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
Information about the ANSYS simulation.
"""

import csv
import pandas as pd
import torch

from wcmi.exception import WCMIError

import wcmi.nn.data

class SimulationInfo():
	"""
	Class whose objects hold ANSYS simulation information such as the number
	and names of input parameters to the simulation algorithm.
	"""

	def __init__(
		self,

		num_sim_inputs,
		num_sim_outputs,

		sim_input_names,
		sim_output_names,

		sim_input_mins=None,
		sim_input_maxs=None,
		sim_output_mins=None,
		sim_output_maxs=None,

		post_update_auto_simplify=True,
	):
		"""
		Create a record of ANSYS simulation information and perform some
		verification on it.
		"""

		# Set properties.
		self._num_sim_inputs  = num_sim_inputs
		self._num_sim_outputs = num_sim_outputs

		self._sim_input_names  = sim_input_names
		self._sim_output_names = sim_output_names

		self._sim_input_mins  = sim_input_mins
		self._sim_input_maxs  = sim_input_maxs
		self._sim_output_mins = sim_output_mins
		self._sim_output_maxs = sim_output_maxs

		# Set parameters.
		self._post_update_auto_simplify = post_update_auto_simplify
		if self._post_update_auto_simplify is None:
			self._post_update_auto_simplify = True

		# Set properties to a consistent type and perform some verification on
		# the properties.
		self.post_update()

	# Property: num_sim_inputs
	@property
	def num_sim_inputs(self):
		return self._num_sim_inputs
	@num_sim_inputs.setter
	def num_sim_inputs(self, num_sim_inputs):
		self._num_sim_inputs = num_sim_inputs
		self.post_update()
	@num_sim_inputs.deleter
	def num_sim_inputs(self):
		del self._num_sim_inputs
		self.post_update()

	# Property: num_sim_outputs
	@property
	def num_sim_outputs(self):
		return self._num_sim_outputs
	@num_sim_outputs.setter
	def num_sim_outputs(self, num_sim_outputs):
		self._num_sim_outputs = num_sim_outputs
		self.post_update()
	@num_sim_outputs.deleter
	def num_sim_outputs(self):
		del self._num_sim_outputs
		self.post_update()

	# Property: sim_input_names
	@property
	def sim_input_names(self):
		return self._sim_input_names
	@sim_input_names.setter
	def sim_input_names(self, sim_input_names):
		self._sim_input_names = sim_input_names
		self.post_update()
	@sim_input_names.deleter
	def sim_input_names(self):
		del self._sim_input_names
		self.post_update()

	# Property: sim_output_names
	@property
	def sim_output_names(self):
		return self._sim_output_names
	@sim_output_names.setter
	def sim_output_names(self, sim_output_names):
		self._sim_output_names = sim_output_names
		self.post_update()
	@sim_output_names.deleter
	def sim_output_names(self):
		del self._sim_output_names
		self.post_update()

	# Property: sim_input_mins
	@property
	def sim_input_mins(self):
		if hasattr(self, "_sim_input_mins"):
			return self._sim_input_mins
		else:
			return None
	@sim_input_mins.setter
	def sim_input_mins(self, sim_input_mins):
		self._sim_input_mins = sim_input_mins
		self.post_update()
	@sim_input_mins.deleter
	def sim_input_mins(self):
		del self._sim_input_mins
		self.post_update()

	# Property: sim_input_maxs
	@property
	def sim_input_maxs(self):
		if hasattr(self, "_sim_input_maxs"):
			return self._sim_input_maxs
		else:
			return None
	@sim_input_maxs.setter
	def sim_input_maxs(self, sim_input_maxs):
		self._sim_input_maxs = sim_input_maxs
		self.post_update()
	@sim_input_maxs.deleter
	def sim_input_maxs(self):
		del self._sim_input_maxs
		self.post_update()

	# Property: sim_output_mins
	@property
	def sim_output_mins(self):
		if hasattr(self, "_sim_output_mins"):
			return self._sim_output_mins
		else:
			return None
	@sim_output_mins.setter
	def sim_output_mins(self, sim_output_mins):
		self._sim_output_mins = sim_output_mins
		self.post_update()
	@sim_output_mins.deleter
	def sim_output_mins(self):
		del self._sim_output_mins
		self.post_update()

	# Property: sim_output_maxs
	@property
	def sim_output_maxs(self):
		if hasattr(self, "_sim_output_maxs"):
			return self._sim_output_maxs
		else:
			return None
	@sim_output_maxs.setter
	def sim_output_maxs(self, sim_output_maxs):
		self._sim_output_maxs = sim_output_maxs
		self.post_update()
	@sim_output_maxs.deleter
	def sim_output_maxs(self):
		del self._sim_output_maxs
		self.post_update()

	# Parameter: post_update_auto_simplify
	@property
	def post_update_auto_simplify(self):
		if hasattr(self, "_post_update_auto_simplify"):
			return self._post_update_auto_simplify
		else:
			return True
	@post_update_auto_simplify.setter
	def post_update_auto_simplify(self, post_update_auto_simplify):
		self._post_update_auto_simplify = post_update_auto_simplify
		self.post_update()
	@post_update_auto_simplify.deleter
	def post_update_auto_simplify(self):
		del self._post_update_auto_simplify
		self.post_update()

	# Pickling and unpickling support for serialization.
	def __getstate__(self):
		"""
		Serialize an instance into a dict for pickle.
		"""

		return {
			"num_sim_inputs":  self.num_sim_inputs,
			"num_sim_outputs": self.num_sim_outputs,

			"sim_input_names":  self.sim_input_names,
			"sim_output_names": self.sim_output_names,

			"sim_input_mins":  self.sim_input_mins,
			"sim_input_maxs":  self.sim_input_maxs,
			"sim_output_mins": self.sim_output_mins,
			"sim_output_maxs": self.sim_output_maxs,

			"post_update_auto_simplify": self.post_update_auto_simplify,
		}

	def __setstate__(self, state):
		"""
		Initialize an instance by deserializing a dict from pickle.
		"""

		# Set properties.
		self._num_sim_inputs  = state["num_sim_inputs"]
		self._num_sim_outputs = state["num_sim_outputs"]

		self._sim_input_names  = state["sim_input_names"]
		self._sim_output_names = state["sim_output_names"]

		self._sim_input_mins  = state["sim_input_mins"]
		self._sim_input_maxs  = state["sim_input_maxs"]
		self._sim_output_mins = state["sim_output_mins"]
		self._sim_output_maxs = state["sim_output_maxs"]

		# Set parameters.
		self._post_update_auto_simplify = state["post_update_auto_simplify"]

		self._post_update_auto_simplify = state["post_update_auto_simplify"]
		if self._post_update_auto_simplify is None:
			self._post_update_auto_simplify = True

		# Set properties to a consistent type and perform some verification on
		# the properties.
		self.post_update()

	def post_update(self):
		"""
		Set the properties to consistent types and encodings and perform some
		verification on them.
		"""
		if self.post_update_auto_simplify:
			self.simplify()
		self.verify()

	@staticmethod
	def none_int_float(val):
		"""
		Static utility method to return None if the value is none; otherwise,
		try constructing an int from a value, and if that fails, then construct
		a float with it, letting exceptions propagate without catching them
		here.
		"""
		if val is None:
			return val
		else:
			try:
				return int(val)
			except ValueError as ex:
				return float(val)

	def simplify(self):
		"""
		Set the properties to have consistent and predictable types.
		"""
		# num_sim_inputs and num_sim_outputs are ints.
		if self._num_sim_inputs is not None:
			self._num_sim_inputs = int(self._num_sim_inputs)
		if self._num_sim_outputs is not None:
			self._num_sim_outputs = int(self._num_sim_outputs)

		# sim_input_names and sim_output_names are tuples of strs.
		if self._sim_input_names is not None:
			if isinstance(self._sim_input_names, str):
				raise WCMIError("error: SimulationInfo.simplify(): self._sim_input_names should be a list, not a string: `{0:s}'.".format(self._sim_input_names))
			self._sim_input_names = (*(str(sim_input_name) for sim_input_name in self._sim_input_names),)
		if self._sim_output_names is not None:
			if isinstance(self._sim_output_names, str):
				raise WCMIError("error: SimulationInfo.simplify(): self._sim_output_names should be a list, not a string: `{0:s}'.".format(self._sim_output_names))
			self._sim_output_names = (*(str(sim_output_name) for sim_output_name in self._sim_output_names),)

		# sim_{input,output}_{mins,maxs} are tuples of ints, floats, or Nones.
		if self._sim_input_mins is not None:
			self._sim_input_mins = (*(self.none_int_float(sim_input_min) for sim_input_min in self._sim_input_mins),)
		if self._sim_input_maxs is not None:
			self._sim_input_maxs = (*(self.none_int_float(sim_input_max) for sim_input_max in self._sim_input_maxs),)
		if self._sim_output_mins is not None:
			self._sim_output_mins = (*(self.none_int_float(sim_output_min) for sim_output_min in self._sim_output_mins),)
		if self._sim_output_maxs is not None:
			self._sim_output_maxs = (*(self.none_int_float(sim_output_max) for sim_output_max in self._sim_output_maxs),)

	def verify(self):
		"""
		Perform some verification on the record and raise an exception if
		issues are detected, like the length of the names not corresponding to
		the number of inputs.
		"""

		# Ensure the object has all needed properties.
		needed_properties=(
			'num_sim_inputs',
			'num_sim_outputs',

			'sim_input_names',
			'sim_output_names',

			# Make the minimums and maximums optional.
			#'sim_input_mins',
			#'sim_input_maxs',
			#'sim_output_mins',
			#'sim_output_maxs',
		)

		for property in needed_properties:
			attr = "_{0:s}".format(property)
			if not hasattr(self, attr):
				raise WCMIError("error: SimulationInfo.verify(): the object is missing the needed property `{0:s}'.".format(property))

		# Make sure the numbers correspond to name list lengths.
		if len(self.sim_input_names) != self.num_sim_inputs:
			raise WCMIError(
				"error: SimulationInfo.verify(): len(self.sim_input_names) != self.num_sim_inputs: {0:d} != {1:s}.".format(
					len(self.sim_input_names), str(self.num_sim_inputs),
				)
			)
		if len(self.sim_output_names) != self.num_sim_outputs:
			raise WCMIError(
				"error: SimulationInfo.verify(): len(self.sim_output_names) != self.num_sim_outputs: {0:d} != {1:s}.".format(
					len(self.sim_output_names), str(self.num_sim_outputs),
				)
			)

		# Make sure the numbers correspond to existing simulation input
		# lengths.
		if self.sim_input_mins is not None:
			if len(self.sim_input_mins) != self.num_sim_inputs:
				raise WCMIError(
					"error: SimulationInfo.verify(): len(self.sim_input_mins) != self.num_sim_inputs: {0:d} != {1:s}.".format(
						len(self.sim_input_mins), str(self.num_sim_inputs),
					)
				)
		if self.sim_input_maxs is not None:
			if len(self.sim_input_maxs) != self.num_sim_inputs:
				raise WCMIError(
					"error: SimulationInfo.verify(): len(self.sim_input_maxs) != self.num_sim_inputs: {0:d} != {1:s}.".format(
						len(self.sim_input_maxs), str(self.num_sim_inputs),
					)
				)
		if self.sim_output_mins is not None:
			if len(self.sim_output_mins) != self.num_sim_outputs:
				raise WCMIError(
					"error: SimulationInfo.verify(): len(self.sim_output_mins) != self.num_sim_outputs: {0:d} != {1:s}.".format(
						len(self.sim_output_mins), str(self.num_sim_outputs)
					)
				)
		if self.sim_output_maxs is not None:
			if len(self.sim_output_maxs) != self.num_sim_outputs:
				raise WCMIError(
					"error: SimulationInfo.verify(): len(self.sim_output_maxs) != self.num_sim_outputs: {0:d} != {1:s}.".format(
						len(self.sim_output_maxs), str(self.num_sim_outputs)
					)
				)

		# Make sure each existent max is >= each existent min.
		if self.sim_input_mins is not None and self.sim_input_maxs is not None:
			for idx, (min, max) in enumerate(zip(self.sim_input_mins, self.sim_input_maxs)):
				name = self.sim_input_names[idx]
				if min is not None and max is not None:
					if not min <= max:
						raise WCMIError(
							"error: SimulationInfo.verify(): simulation input value #{0:d}/#{1:d} (`{2:s}') has a minimum greater than the maximum: {3:d} > {4:d}.".format(
								idx + 1,
								self.num_sim_inputs,
								name,
								min,
								max,
							)
						)
		if self.sim_output_mins is not None and self.sim_output_maxs is not None:
			for idx, (min, max) in enumerate(zip(self.sim_output_mins, self.sim_output_maxs)):
				name = self.sim_output_names[idx]
				if min is not None and max is not None:
					if not min <= max:
						raise WCMIError(
							"error: SimulationInfo.verify(): simulation output value #{0:d}/#{1:d} (`{2:s}') has a minimum greater than the maximum: {3:d} > {4:d}.".format(
								idx + 1,
								self.num_sim_outputs,
								name,
								min,
								max,
							)
						)

	def get_sim_input_ranges(self, if_either_none_then_both_none=False):
		"""
		Returns a tuple of pairs of input range bounds, which might be null.

		E.g. simulation_info.get_sim_input_ranges() == ((1, 100), (None, 8), (None, None),)

		To return ((1, 100), (None, None), (None, None)) rather than the
		previous example, pass True for if_either_none_then_both_none.
		"""

		# Set default values.
		if if_either_none_then_both_none is None:
			if_either_none_then_both_none = False

		# Get and return the ranges.
		if self.sim_input_mins is not None:
			expanded_mins = self.sim_input_mins
		else:
			expanded_mins = self.num_sim_inputs * (None,)

		if self.sim_input_maxs is not None:
			expanded_maxs = self.sim_input_maxs
		else:
			expanded_maxs = self.num_sim_inputs * (None,)

		if if_either_none_then_both_none:
			return (*((min, max) if min is not None and max is not None else (None, None) for min, max in zip(expanded_mins, expanded_maxs)),)
		else:
			return (*((min, max) for min, max in zip(expanded_mins, expanded_maxs)),)

	def get_sim_output_ranges(self, if_either_none_then_both_none=False):
		"""
		Returns a tuple of pairs of output range bounds, which might be null.

		E.g. simulation_info.get_sim_output_ranges() == ((1, 100), (None, 8), (None, None),)

		To return ((1, 100), (None, None), (None, None)) rather than the
		previous example, pass True for if_either_none_then_both_none.
		"""

		# Set default values.
		if if_either_none_then_both_none is None:
			if_either_none_then_both_none = False

		# Get and return the ranges.
		if self.sim_output_mins is not None:
			expanded_mins = self.sim_output_mins
		else:
			expanded_mins = self.num_sim_outputs * (None,)

		if self.sim_output_maxs is not None:
			expanded_maxs = self.sim_output_maxs
		else:
			expanded_maxs = self.num_sim_outputs * (None,)

		if if_either_none_then_both_none:
			return (*((min, max) if min is not None and max is not None else (None, None) for min, max in zip(expanded_mins, expanded_maxs)),)
		else:
			return (*((min, max) for min, max in zip(expanded_mins, expanded_maxs)),)

def get_simulation_info():
	return SimulationInfo(
		num_sim_inputs=7,
		num_sim_outputs=5,

		sim_input_names=(
			"Iin[A]",   # [1, 100]
			"Iout[A]",  # [1, 100]
			"l[mm]",    # [3, 1500]
			"p1[mm]",   # [3, 150]
			"p2[mm]",   # [3, 150]
			"p3[mm]",   # [3, 150]
			"win[mm]",  # [1, 1000]
		),
		sim_output_names=(
			"kdiff[%]",        # [0.1, 40]
			"Bleak[uT]",       # [1,   300]
			"V_PriWind[cm3]",  # [1,   300]
			"V_PriCore[cm3]",  # [1,   10000]
			"Pout[W]",         # [1,   3000]
		),

		sim_input_mins=(
			1, 1, 3, 3, 3, 3, 1,
		),
		sim_input_maxs=(
			100, 100, 1500, 150, 150, 150, 1000,
		),
		sim_output_mins=(
			0.1, 1, 1, 1, 1
		),
		sim_output_maxs=(
			40, 300, 300, 10000, 3000,
		),
	)

# The simulation information.
simulation_info = get_simulation_info()

class SimulationData():
	"""
	Container of a pandas frame loaded from CSV data.

	It can be access by e.g. `simulation_data.data`.
	"""
	def __init__(self, load_data_path, save_data_path=None, verify_gan_n=True, optional_gan_n=True, gan_n=None, simulation_info=None, *args, **kwargs):
		"""
		Initialize a SimulationData by loading data from a CSV file.

		If the CSV data is expected to have n GAN columns, set gan_n to n (or
		None for default_gan_n) and verify_gan_n to True.  Otherwise, set
		verify_gan_n to False.

		If optional_gan_n is True and verify_gan_n is True, then either there
		can be gan_n GAN columns or zero GAN columns, but no other number of
		GAN columns.  The default value for optional_gan_n is True.
		"""
		super().__init__(*args, **kwargs)

		# Default arguments.
		if simulation_info is None:
			simulation_info = get_simulation_info()
		if gan_n is None:
			gan_n = wcmi.nn.data.default_gan_n

		# Set attributes.
		self.load_data_path  = load_data_path
		self.save_data_path  = save_data_path
		self.verify_gan_n    = verify_gan_n
		self.optional_gan_n  = optional_gan_n
		self.gan_n           = gan_n
		self.simulation_info = simulation_info

		# Set other attributes.
		data = None  # Will be set in load().

		# Load the CSV file.
		self.load(load_data_path)

	def load(self, load_data_path=None, verify_gan_n=None, optional_gan_n=None, gan_n=None):
		"""
		Set self.data to a pandas frame containing the CSV data.

		12 columns plus n GAN columns are expected.

		If the CSV data is expected to have n GAN columns, set gan_n to n (or
		None for default_gan_n) and verify_gan_n to True.  Otherwise, set
		verify_gan_n to False.

		If optional_gan_n is True and verify_gan_n is True, then either there
		can be gan_n GAN columns or zero GAN columns, but no other number of
		GAN columns.  The default value for optional_gan_n is True.
		"""

		# Default arguments.
		if load_data_path is None:
			load_data_path = self.load_data_path
		if verify_gan_n is None:
			verify_gan_n = self.verify_gan_n
		if optional_gan_n is None:
			optional_gan_n = self.optional_gan_n
		if gan_n is None:
			gan_n = self.gan_n
		if gan_n is None:
			gan_n = wcmi.nn.data.default_gan_n

		# Read the CSV file.
		self.data = pd.read_csv(load_data_path)

		# Fail if there are too few columns (< 12).
		num_csv_columns = len(self.data.columns)
		min_needed_columns = self.simulation_info.num_sim_inputs + self.simulation_info.num_sim_outputs
		if num_csv_columns < min_needed_columns:
			raise WCMIError("error: SimulationData.load(): there are fewer columns than the number needed: {0:d} < {1:d}".format(num_csv_columns, min_needed_columns))

		# Fail on any unrecognized columns.
		valid_names = self.simulation_info.sim_input_names + self.simulation_info.sim_output_names
		column_names = self.data.columns.values
		liberal_valid_names  = [name.lower().strip() for name in valid_names]
		liberal_column_names = [name.lower().strip() for name in column_names]
		for column_name in self.data.columns.values:
			liberal_column_name = column_name.lower().strip()
			if not column_name.lower().strip().startswith("GAN".lower().strip()) and liberal_column_name not in liberal_valid_names:
				err_msg = "\n".join((
					"error: SimulationData.load(): unrecognized column name in CSV file: {0:s}".format(
						column_name,
					),
					"",
					"Valid column names:",
					*("  - {0:s}".format(repr(sim_input_name)) for sim_input_name in self.sim_input_names),
				))
				raise WCMIError(err_msg)

		# If verify_gan_n is provided (as an int), make sure we have 12 + verify_gan_n columns.
		if verify_gan_n:
			num_needed_columns = self.simulation_info.num_sim_inputs + self.simulation_info.num_sim_outputs + gan_n
			if num_csv_columns != num_needed_columns:
				# Invalid number of columns, but first check for
				# optional_gan_n.
				if not (optional_gan_n and num_csv_columns == min_needed_columns):
					num_csv_gan_columns = num_csv_columns - min_needed_columns
					raise WCMIError("error: SimulationData.load(): the number of GAN columns does not equal what was expected: {0:d} != {1:d}".format(num_csv_gan_columns, gan_n))

		# Verification is done.  Now just rearrange the columns to match the
		# order provided by self.simulation_input_names then
		# self.simulation_output_names if there is a difference.
		# TODO: clean up a bit; very long lines.
		new_column_names = sorted([name for name in column_names if name.lower().strip() not in liberal_column_names], key=lambda: liberal_column_names.index()) + [name for name in column_names if name.lower().strip() in liberal_column_names]
		self.data.reindex(new_column_names)

	def save(self, data=None, save_data_path=None, verify_gan_n=None, optional_gan_n=None, gan_n=None):
		"""
		Write self.data, a pandas frame, to a CSV file.

		It is expected to have 19 + n>=0 columns (i.e. self.num_sim_outputs
		additional columns for predictions.)
		"""

		# Default arguments.
		if data is None:
			data = self.data
		if save_data_path is None:
			save_data_path = self.save_data_path
		if verify_gan_n is None:
			verify_gan_n = self.verify_gan_n
		if optional_gan_n is None:
			optional_gan_n = self.optional_gan_n
		if gan_n is None:
			gan_n = self.gan_n
		if gan_n is None:
			gan_n = wcmi.nn.data.default_gan_n

		# Ensure we have somewhere to write to.
		if save_data_path is None:
			raise WCMIError("error: SimulationData.save(): no save path is available!")

		# Fail if there are too few columns (< 19).
		num_csv_columns = len(data.columns)
		min_needed_columns = self.simulation_info.num_sim_inputs + self.simulation_info.num_sim_outputs + self.simulation_info.num_sim_inputs
		if num_csv_columns < min_needed_columns:
			raise WCMIError("error: SimulationData.save(): there are fewer columns than the number needed: {0:d} < {1:d}".format(num_csv_columns, min_needed_columns))

		# If verify_gan_n is enabled, verify the exact number of GAN columns.
		if verify_gan_n:
			num_needed_columns = min_needed_columns + gan_n
			if num_csv_columns != num_needed_columns:
				# Invalid number of columns, but first check for
				# optional_gan_n.
				if not (optional_gan_n and num_csv_columns == min_needed_columns):
					num_csv_gan_columns = num_csv_columns - min_needed_columns
					raise WCMIError("error: SimulationData.save(): the number of GAN columns does not equal what was expected: {0:d} != {1:d}".format(num_csv_gan_columns, gan_n))

		# Write the CSV file.
		# (index=False to omit the extra ID column.)
		data.to_csv(save_data_path, index=False)
