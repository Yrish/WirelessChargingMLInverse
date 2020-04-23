# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
Information about the ANSYS simulation.
"""

from wcmi.exception import WCMIError

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

		self._post_update_auto_simplify = post_update_auto_simplify
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
				raise WCMIError("error: simulation.simplify(): self._sim_input_names should be a list, not a string: `{0:s}'.".format(self._sim_input_names))
			self._sim_input_names = (*(str(sim_input_name) for sim_input_name in self._sim_input_names),)
		if self._sim_output_names is not None:
			if isinstance(self._sim_output_names, str):
				raise WCMIError("error: simulation.simplify(): self._sim_output_names should be a list, not a string: `{0:s}'.".format(self._sim_output_names))
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
				raise WCMIError("error: simulation.verify(): the object is missing the needed property `{0:s}'.".format(property))

		# Make sure the numbers correspond to name list lengths.
		if len(self.sim_input_names) != self.num_sim_inputs:
			raise WCMIError(
				"error: simulation.verify(): len(self.sim_input_names) != self.num_sim_inputs: {0:d} != {1:s}.".format(
					len(self.sim_input_names), str(self.num_sim_inputs),
				)
			)
		if len(self.sim_output_names) != self.num_sim_outputs:
			raise WCMIError(
				"error: simulation.verify(): len(self.sim_output_names) != self.num_sim_outputs: {0:d} != {1:s}.".format(
					len(self.sim_output_names), str(self.num_sim_outputs),
				)
			)

		# Make sure the numbers correspond to existing simulation input
		# lengths.
		if self.sim_input_mins is not None:
			if len(self.sim_input_mins) != self.num_sim_inputs:
				raise WCMIError(
					"error: simulation.verify(): len(self.sim_input_mins) != self.num_sim_inputs: {0:d} != {1:s}.".format(
						len(self.sim_input_mins), str(self.num_sim_inputs),
					)
				)
		if self.sim_input_maxs is not None:
			if len(self.sim_input_maxs) != self.num_sim_inputs:
				raise WCMIError(
					"error: simulation.verify(): len(self.sim_input_maxs) != self.num_sim_inputs: {0:d} != {1:s}.".format(
						len(self.sim_input_maxs), str(self.num_sim_inputs),
					)
				)
		if self.sim_output_mins is not None:
			if len(self.sim_output_mins) != self.num_sim_outputs:
				raise WCMIError(
					"error: simulation.verify(): len(self.sim_output_mins) != self.num_sim_outputs: {0:d} != {1:s}.".format(
						len(self.sim_output_mins), str(self.num_sim_outputs)
					)
				)
		if self.sim_output_maxs is not None:
			if len(self.sim_output_maxs) != self.num_sim_outputs:
				raise WCMIError(
					"error: simulation.verify(): len(self.sim_output_maxs) != self.num_sim_outputs: {0:d} != {1:s}.".format(
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
							"error: simulation.verify(): simulation input value #{0:d}/#{1:d} (`{2:s}') has a minimum greater than the maximum: {3:d} > {4:d}.".format(
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
							"error: simulation.verify(): simulation output value #{0:d}/#{1:d} (`{2:s}') has a minimum greater than the maximum: {3:d} > {4:d}.".format(
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
