# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
Module providing methods to interface with the neural networks provided by this
package.
"""

from wcmi.exception import WCMIError

import wcmi.nn as nn
import wcmi.nn.dense as dense
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

	# Load the model.
	#
	# Optionally, the model might be randomly initialized if it hasn't been
	# trained before.
	mdl        = gan.GAN          if use_gan else dense.Dense
	mdl_kwargs = {'gan_n': gan_n} if use_gan else {}
	model = mdl(
		load_model_path=load_model_path,
		save_model_path=None,
		auto_load_model=True,
		**kwargs,
	)

	# TODO
	print("(To be implemented...)")
	pass
	return

	# Save the trained model.
	model.save()

def run(use_gan=True, load_model_path=None, load_data_path=None, save_data_path=None, gan_n=gan.default_gan_n):
	"""
	Load the CSV data, pass it through the neural network, and write a new CSV
	file that includes what the neural network predicted.

	Run the application with --help for more information on the structure of
	the loaded and written CSV files.
	"""
	# Default arguments.
	if gan_n is None:
		gan_n = gan.default_gan_n

	# Argument verification.
	if load_data_path is None:
		raise WCMIError("error: run requires --load-data=.../path/to/data.csv to be specified.")
	if save_data_path is None:
		raise WCMIError("error: run requires --save-data=.../path/to/data.csv to be specified.")
	if load_model_path is None:
		raise WCMIError("error: run requires --load-model=.../path/to/model.pt to be specified.")

	# Read the CSV file.
	simulation_data = simulation.SimulationData(
		load_data_path=load_data_path,
		save_data_path=save_data_path,
		verify_gan_n=True,
		optional_gan_n=True,
		gan_n=gan_n,
		simulation_info=simulation.simulation_info,
	)

	# Load the model.
	mdl        = gan.GAN          if use_gan else dense.Dense
	mdl_kwargs = {'gan_n': gan_n} if use_gan else {}
	model = mdl(
		load_model_path=load_model_path,
		save_model_path=None,
		auto_load_model=True,
		**mdl_kwargs,
	)

	# Feed the data to the model and collect the output.
	num_sim_in_columns     = simulation_data.simulation_info.num_nim_inputs
	num_sim_in_out_columns = num_sim_in_out_columns + simulation_data.simulation_info.num_nim_outputs

	## Pass the numpy array through the model.
	npoutput = model(simulation_data.values[:,:num_sim_in_columns])

	## Reconstruct the Pandas frame with appropriate columns.
	input_columns = self.data.columns.values.tolist()

	predicted_columns = ["pred_{0:s}".format(name) for name in input_columns[:num_sim_in_columns]]

	output_columns = input_columns[:]
	output_columns[num_sim_in_out_columns:num_sim_in_out_columns] = predicted_columns[:]

	if use_gan:
		# If the input columns lacked GAN columns, then add them now, since the
		# GAN columns are present.
		if len(input_columns) <= num_sim_in_columns:
			# No GAN columns.  Add them.
			output_columns += ["GAN_{0:d}".format(gan_column_num) for gan_column_num in range(gan_n)]

	output = pandas.DataFrame(
		data=npoutput,
		columns=output_columns,
	)

	# Write the output.
	simulation_data.save(output)

def stats():
	"""
	(To be documented...)
	"""
	print("(To be implemented...)")
	pass
