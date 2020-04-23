# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
Module providing methods to interface with the neural networks provided by this
package.
"""

from wcmi.exception import WCMIError

import wcmi.nn.gan as gan
import wcmi.simulation as simulation

def train(use_gan=True, load_model_path=None, save_model_path=None, load_data_path=None, gan_n=gan.default_gan_n):
	"""
	(To be documented...)
	"""

	# Default arguments.
	if gan_n is None:
		gan_n = gan.default_gan_n

	# Argument verification.
	if load_data_path is None:
		raise WCMIError("error: train requires --load-data=.../path/to/data.csv to be specified.")

	# Read the CSV file.
	simulation_data = simulation.SimulationData(
		load_data_path=load_data_path,
		save_data_path=save_data_path,
		verify_gan_n=True,
		optional_gan_n=True,
		gan_n=gan_n,
		simulation_info=simulation.simulation_info,
	)

	print("(To be implemented...)")
	pass

def run(use_gan=True, load_model_path=None, save_model_path=None, load_data_path=None, save_data_path=None, gan_n=gan.default_gan_n):
	"""
	(To be documented...)
	"""
	# Default arguments.
	if gan_n is None:
		gan_n = gan.default_gan_n

	# Argument verification.
	if load_data_path is None:
		raise WCMIError("error: train requires --load-data=.../path/to/data.csv to be specified.")

	# Read the CSV file.
	simulation_data = simulation.SimulationData(
		load_data_path=load_data_path,
		save_data_path=save_data_path,
		verify_gan_n=True,
		optional_gan_n=True,
		gan_n=gan_n,
		simulation_info=simulation.simulation_info,
	)

	print("(To be implemented...)")
	pass

def stats():
	"""
	(To be documented...)
	"""
	print("(To be implemented...)")
	pass
