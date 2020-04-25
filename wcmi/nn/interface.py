# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
Module providing methods to interface with the neural networks provided by this
package.
"""

import pandas as pd
import math
import numpy as np
import os
import shutil
import torch
import torch.nn as nn

from wcmi.exception import WCMIError

import wcmi.nn as wnn
import wcmi.nn.data as data
import wcmi.nn.dense as dense
import wcmi.nn.gan as gan
import wcmi.simulation as simulation

def train(
	use_gan=True, load_model_path=None, save_model_path=None,
	load_data_path=None, save_data_path=None, gan_n=gan.default_gan_n,
	num_epochs=data.default_num_epochs,
	status_every_epoch=data.default_status_every_epoch,
	status_every_sample=data.default_status_every_sample,
	batch_size=data.default_batch_size,
):
	"""
	Train a neural network with data and save it.
	"""

	# Default arguments.
	if gan_n is None:
		gan_n = gan.default_gan_n

	# Argument verification.
	if load_data_path is None:
		raise WCMIError("error: train requires --load-data=.../path/to/data.csv to be specified.")
	if save_model_path is None:
		raise WCMIError("error: train requires --save-model.../path/to/model.pt to be specified.")
	if num_epochs < 1:
		raise WCMIError("error: train requires --num-epochs to be at least 1.")

	# Read the CSV file.
	simulation_data = simulation.SimulationData(
		load_data_path=load_data_path,
		save_data_path=None,  # (This is for CSV prediction output, not MSE data.  Set to None.)
		verify_gan_n=True,
		optional_gan_n=True,
		gan_n=gan_n,
		simulation_info=simulation.simulation_info,
	)

	# Data verification.
	if len(simulation_data.data) <= 0:
		raise WCMIError("error: train requires at least one sample in the CSV file.")

	# Calculate sizes, numbers, and lengths.
	num_samples = len(simulation_data.data)
	num_testing_samples = int(round(data.test_proportion * num_samples))  # (redundant int())
	num_training_samples = num_samples - num_testing_samples

	if batch_size <= 0:
		batch_size = num_samples

	num_batches = num_samples // batch_size
	final_batch_size = num_samples % batch_size
	if final_batch_size == 0:
		final_batch_size = batch_size

	num_training_batches = num_training_samples // batch_size
	final_training_batch_size = num_training_samples % batch_size
	if final_training_batch_size == 0:
		final_training_batch_size = training_batch_size

	num_testing_batches = num_testing_samples // batch_size
	final_testing_batch_size = num_testing_samples % batch_size
	if final_testing_batch_size == 0:
		final_testing_batch_size = testing_batch_size

	# Get the input and labels (target).
	num_sim_in_columns     = simulation_data.simulation_info.num_sim_inputs
	num_sim_in_out_columns = num_sim_in_columns + simulation_data.simulation_info.num_sim_outputs

	#npdata = simulation_data.data.values[:, :num_sim_in_out_columns]  # (No need for a numpy copy.)
	all_data = torch.Tensor(simulation_data.data.values[:, :num_sim_in_out_columns]).to(data.device)
	all_labels = all_data.view(all_data.shape)[:, :num_sim_in_columns]
	all_input  = all_data.view(all_data.shape)[:, num_sim_in_columns:num_sim_in_out_columns]

	# Get mean, stddev, min, and max of each input and label column for
	# standardization or normalization.
	all_nplabels  = all_labels.numpy()
	all_npinput   = all_input.numpy()
	label_means = torch.tensor(np.apply_along_axis(np.mean, axis=0, arr=all_nplabels))
	label_stds  = torch.tensor(np.apply_along_axis(np.std,  axis=0, arr=all_nplabels))
	label_mins  = torch.tensor(np.apply_along_axis(np.min,  axis=0, arr=all_nplabels))
	label_maxs  = torch.tensor(np.apply_along_axis(np.max,  axis=0, arr=all_nplabels))
	input_means = torch.tensor(np.apply_along_axis(np.mean, axis=0, arr=all_npinput))
	input_stds  = torch.tensor(np.apply_along_axis(np.std,  axis=0, arr=all_npinput))
	input_mins  = torch.tensor(np.apply_along_axis(np.min,  axis=0, arr=all_npinput))
	input_maxs  = torch.tensor(np.apply_along_axis(np.max,  axis=0, arr=all_npinput))

	# Load the model.
	#
	# Optionally, the model might be randomly initialized if it hasn't been
	# trained before.
	mdl        = gan.GAN          if use_gan else dense.Dense
	mdl_kwargs = {'gan_n': gan_n} if use_gan else {}
	model = mdl(
		load_model_path=load_model_path,
		save_model_path=save_model_path,
		auto_load_model=True,
		population_mean_in=input_means,
		population_std_in=input_stds,
		population_min_in=input_mins,
		population_max_in=input_maxs,
		population_mean_out=label_means,
		population_std_out=label_stds,
		population_min_out=label_mins,
		population_max_out=label_maxs,
		**mdl_kwargs,
	)
	# If CUDA is available, move the model to the GPU.
	model = model.to(data.device)

	# Split data into training data and test data.  The test data will be
	# invisible during the training (except to report accuracies).

	# Set a reproducible initial seed for a reproducible split, but then
	# reset the seed after the split.
	torch.random.manual_seed(data.testing_split_seed)

	# Shuffle the rows of data.
	#np.random.shuffle(npdata)
	# c.f. https://stackoverflow.com/a/53284632
	all_data = all_data[torch.randperm(all_data.size()[0])].to(data.device)

	# Restore randomness.
	#torch.random.manual_seed(torch.random.seed())
	# Fix an error, c.f.
	# https://discuss.pytorch.org/t/initial-seed-too-large/28832
	torch.random.manual_seed(torch.random.seed() & ((1<<63)-1))

	testing_data = all_data.view(all_data.shape)[:num_testing_samples]
	training_data = all_data.view(all_data.shape)[num_testing_samples:]

	testing_labels = testing_data.view(testing_data.shape)[:, :num_sim_in_columns]
	testing_input  = testing_data.view(testing_data.shape)[:, num_sim_in_columns:num_sim_in_out_columns]

	training_labels = training_data.view(training_data.shape)[:, :num_sim_in_columns]
	training_input  = training_data.view(training_data.shape)[:, num_sim_in_columns:num_sim_in_out_columns]

	# Train the model.
	if not use_gan:
		# Get a tensor to store predictions for each epoch.  It will be
		# overwritten at each epoch.
		current_epoch_testing_errors = torch.zeros(testing_labels.shape, device=data.device, requires_grad=False)
		current_epoch_training_errors = torch.zeros(training_labels.shape, device=data.device, requires_grad=False)

		# After each epoch, set the corresponding element in this array to the
		# calculated MSE accuracy.
		epoch_training_mse = torch.zeros((num_epochs,num_sim_in_columns,), device=data.device)
		epoch_testing_mse = torch.zeros((num_epochs,num_sim_in_columns,), device=data.device)

		# Define the loss function and the optimizer.
		loss_function = nn.MSELoss()

		# Give the optimizer a reference to our model's parameters, which
		# includes the models weights and biases.  The optimizer will update
		# them.
		optimizer = torch.optim.SGD(
			model.parameters(),
			lr=data.learning_rate,
			momentum=data.momentum,
			weight_decay=data.weight_decay,
			dampening=data.dampening,
			nesterov=data.nesterov,
		)

		# Let the user know on which device training is occurring.
		print("device: {0:s}".format(str(data.device)))

		# Run all epochs.
		for epoch in range(num_epochs):
			# Should we print a status update?
			if status_every_epoch <= 0:
				status_enabled = False
			else:
				status_enabled = epoch % status_every_epoch == 0

			if status_enabled:
				#if epoch > 1:
				#	print("")
				print("")
				print("Beginning epoch #{0:d}/{1:d}.".format(epoch + 1, num_epochs))

			# Shuffle the rows of data.
			training_data = training_data[torch.randperm(training_data.size()[0])].to(data.device)

			# Clear the errors tensors for this epoch.
			current_epoch_testing_errors = torch.zeros(testing_labels.shape, out=current_epoch_testing_errors, device=data.device, requires_grad=False)
			current_epoch_training_errors = torch.zeros(training_labels.shape, out=current_epoch_training_errors, device=data.device, requires_grad=False)

			# Zero the gradient.
			optimizer.zero_grad()

			# Run all batches in the epoch.
			for batch in range(num_training_batches):
				if not status_enabled or status_every_sample <= 0:
					substatus_enabled = False
				else:
					# Example:
					# 	0*2=0  % 4=0  < 2
					# 	1*2=2  % 4=2 !< 2
					# 	2*2=4  % 4=0  < 2
					# 	3*2=6  % 4=2 !< 2
					# 	4*2=8  % 4=0  < 2
					# 	5*2=10 % 4=2 !< 2
					substatus_enabled = batch * batch_size % status_every_sample < batch_size

				# Print a status for the next sample?
				if substatus_enabled:
					print("  Beginning sample #{0:d}/{1:d} (epoch #{2:d}/{3:d}).".format(
						batch * batch_size + 1,
						num_samples,
						epoch + 1,
						num_epochs,
					))

				# Get this batch of samples.
				batch_slice = slice(batch * batch_size, (batch + 1) * batch_size)  # i.e. [batch * batch_size:(batch + 1) * batch_size]

				#batch_data   = training_data[batch_slice]
				batch_input  = training_input[batch_slice]
				batch_labels = training_labels[batch_slice]

				# Forward pass.
				batch_output = model(batch_input)
				loss = loss_function(batch_output, batch_labels)

				# Record the errors for this batch.
				current_epoch_training_errors[batch_slice] = batch_output.detach() - batch_labels.detach()

				# Backpropogate to calculate the gradient and then optimize to
				# update the weights (parameters).
				loss.backward()
				optimizer.step()

			# Calculate the MSE for each prediction column (7-element vector),
			# then assign it to epoch_mse_errors
			current_epoch_training_mse = (current_epoch_training_errors**2).mean(0)
			epoch_training_mse[epoch] = current_epoch_training_mse
			current_epoch_training_mse_norm = current_epoch_training_mse.norm()

			# Perform testing for this epoch.
			#
			# Disable gradient calculation during this phase with
			# torch.no_grad() since we're not doing backpropagation here.
			with torch.no_grad():
				# Now run the test batches.
				for batch in range(num_testing_batches):
					total_batch = batch + num_training_batches

					if not status_enabled or status_every_sample <= 0:
						substatus_enabled = False
					else:
						# Example:
						# 	0*2=0  % 4=0  < 2
						# 	1*2=2  % 4=2 !< 2
						# 	2*2=4  % 4=0  < 2
						# 	3*2=6  % 4=2 !< 2
						# 	4*2=8  % 4=0  < 2
						# 	5*2=10 % 4=2 !< 2
						substatus_enabled = total_batch * batch_size % status_every_sample < batch_size

					# Print a status for the next sample?
					if substatus_enabled:
						print("  Beginning sample #{0:d}/{1:d} (testing phase) (epoch #{2:d}/{3:d}).".format(
							total_batch * batch_size + 1,
							num_samples,
							epoch + 1,
							num_epochs,
						))

					# Get this batch of samples.
					batch_slice = slice(batch * batch_size, (batch + 1) * batch_size)  # i.e. [batch * batch_size:(batch + 1) * batch_size]

					#batch_data   = testing_data[batch_slice]
					batch_input  = testing_input[batch_slice]
					batch_labels = testing_labels[batch_slice]

					# Forward pass.
					batch_output = model(batch_input)
					loss = loss_function(batch_output, batch_labels)

					# Record the errors for this batch.
					current_epoch_testing_errors[batch_slice] = batch_output.detach() - batch_labels.detach()

				# Calculate the MSE for each prediction column (7-element vector),
				# then assign it to epoch_mse_errors
				current_epoch_testing_mse = (current_epoch_testing_errors**2).mean(0)
				epoch_testing_mse[epoch] = current_epoch_testing_mse
				current_epoch_testing_mse_norm = current_epoch_testing_mse.norm()

			# Unless we're outputting a status message every epoch, let the
			# user know we've finished this epoch.
			#if status_enabled and status_every_epoch > 1:
			if status_enabled:
				print(
					"Done training epoch #{0:d}/{1:d} (testing vs. training MSE norm: {2:f} vs. {3:f} (lower is more accurate)).".format(
						epoch + 1, num_epochs,
						current_epoch_testing_mse_norm,
						current_epoch_training_mse_norm,
					)
				)

		# We are done training the Dense neural network.
		#
		# Get and print some stats.
		last_testing_mse = epoch_testing_mse[-1]
		last_training_mse = epoch_training_mse[-1]

		all_nplabels = all_labels.numpy()

		print("")
		print("Done training last epoch.  Preparing statistics...")

		def stat_format(fmt, tvec=None, lvec=None, float_str_min_len=15):
			"""
			Print a formatting line.  Specify tvec for a tensor vector (norm
			added) or lvec for a list of strings.
			"""
			if tvec is not None:
				return fmt.format(str(float_str_min_len)).format(
					"<{0:s}>".format(", ".join(["{{0:{0:s}f}}".format(str(float_str_min_len)).format(component) for component in tvec])),
					tvec.norm(),
				)
			elif lvec is not None:
				return fmt.format(str(float_str_min_len)).format(
					"<{0:s}>".format(", ".join(["{{0:>{0:s}s}}".format(str(float_str_min_len)).format(str(component)) for component in lvec])),
				)
			else:
				return fmt

		bold = "\033[1m"
		white = "\033[37m"
		clear = "\033[0;0m"
		#b = bold + white
		b = white
		e = clear
		stat_fmts = (
			("", None, None),
			("Last testing MSE   (norm) : {{0:s}} ({{1:{0:s}f}})", last_testing_mse, None),
			(b+"Last testing RMSE  (norm) : {{0:s}} ({{1:{0:s}f}})"+e, last_testing_mse.sqrt(), None),
			("Last training MSE  (norm) : {{0:s}} ({{1:{0:s}f}})", last_training_mse, None),
			("Last training RMSE (norm) : {{0:s}} ({{1:{0:s}f}})", last_training_mse.sqrt(), None),
			("", None, None),
			("Label column names        : {{0:s}}", None, simulation_data.simulation_info.sim_input_names),
			("", None, None),
			("All labels mean    (norm) : {{0:s}} ({{1:{0:s}f}})", all_labels.mean(0), None),
			("All labels var     (norm) : {{0:s}} ({{1:{0:s}f}})", all_labels.var(0), None),
			(b+"All labels stddev  (norm) : {{0:s}} ({{1:{0:s}f}})"+e, all_labels.std(0), None),
			("", None, None),
			("All labels min     (norm) : {{0:s}} ({{1:{0:s}f}})", torch.Tensor(np.quantile(all_nplabels, 0, 0)), None),
			("...1st quartile    (norm) : {{0:s}} ({{1:{0:s}f}})", torch.Tensor(np.quantile(all_nplabels, 0.25, 0)), None),
			("All labels median  (norm) : {{0:s}} ({{1:{0:s}f}})", torch.Tensor(np.quantile(all_nplabels, 0.5, 0)), None),
			("...3rd quartile    (norm) : {{0:s}} ({{1:{0:s}f}})", torch.Tensor(np.quantile(all_nplabels, 0.75, 0)), None),
			("All labels max     (norm) : {{0:s}} ({{1:{0:s}f}})", torch.Tensor(np.quantile(all_nplabels, 1, 0)), None),
		)

		def stat_fmt_lines(float_str_min_line):
			"""Given a float_str_min_line value, return formatted stats lines."""
			return (*(
				stat_format(*vals, float_str_min_len) for vals in stat_fmts
			),)
		def print_stat_fmt_lines(float_str_min_line):
			"""Given a float_str_min_line value, print formatted stats lines."""
			for line in stat_fmt_lines(float_str_min_line):
				print(line)

		# Start at 15 and decrease until the maximimum line length is <=
		# COLUMNS, then keep decreasing until the number of maximum lengthed
		# lines changes.
		#float_str_min_len = 15
		float_str_min_len = 12
		if data.auto_size_formatting:
			cols = None
			columns, rows = shutil.get_terminal_size((1, 1))
			if columns > 1:
				cols = columns
			if cols is not None:
				# We know the columns, so we can be more liberal in the number we
				# start with.  Start at 30.
				float_str_min_len = 30

			last_max_line_len_count = None
			# Try decreasing to 0, inclusive.
			for try_float_str_min_len in range(float_str_min_len, -1, -1):
				lines = stat_fmt_lines(try_float_str_min_len)
				max_line_len = max([len(line) for line in lines])
				max_line_len_count = len([line for line in lines if len(line) >= max_line_len])
				if cols is None or max_line_len_count <= cols:
					if last_max_line_len_count is not None and max_line_len_count < last_max_line_len_count:
						break
				last_max_line_len_count = max_line_len_count
				float_str_min_len = try_float_str_min_len

		# Now print the stats.
		print_stat_fmt_lines(float_str_min_len)

		# Did the user specify to save MSE errors?
		if save_data_path is not None:
			mse_columns = ["is_training"] + ["mse_{0:s}".format(column) for column in simulation_data.simulation_info.sim_input_names]
			# Prepend the "is_training" column as the first.
			epoch_training_np_mse = epoch_training_mse.numpy()
			epoch_testing_np_mse = epoch_testing_mse.numpy()
			testing_mse = np.concatenate(
				(
					np.zeros((num_epochs,1,)),
					epoch_testing_np_mse,
				),
				axis=1,
			)
			training_mse = np.concatenate(
				(
					np.ones((num_epochs,1,)),
					epoch_training_np_mse,
				),
				axis=1,
			)
			mse = np.concatenate(
				(
					testing_mse,
					training_mse,
				),
				axis=0,
			)
			mse_output = pd.DataFrame(
				data=mse,
				columns=mse_columns,
			)
			# c.f. https://stackoverflow.com/a/41591077
			mse_output["is_training"] = mse_output["is_training"].astype(int)
			mse_output.to_csv(save_data_path, index=False)

			print("")
			print("Wrote MSE errors (testing MSE for each epoch and then training MSE for each epoch) `{0:s}'.".format(save_data_path))

		# We're done.  Catch you later.
		print("")
		print("Done training all epochs.")
		print("Have a good day.")

	else:
		# TODO
		print("(To be implemented...)")
		raise NotImplementedError("error: train: the train action is not yet implemented for --gan.")
		pass
		return

	# Save the trained model.
	model.save()

def run(use_gan=True, load_model_path=None, load_data_path=None, save_data_path=None, gan_n=gan.default_gan_n, output_keep_out_of_bounds=False):
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

	# Ensure there is at least one sample.
	if len(simulation_data.data) <= 0:
		raise WCMIError("error: run requires the CSV data loaded to contain at least one sample.")

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
	num_sim_in_columns     = simulation_data.simulation_info.num_sim_inputs
	num_sim_in_out_columns = num_sim_in_columns + simulation_data.simulation_info.num_sim_outputs

	#npdata = simulation_data.data.values[:, :num_sim_in_out_columns]  # (No need for a numpy copy.)
	all_data = torch.tensor(simulation_data.data.values[:, :num_sim_in_out_columns], dtype=torch.float32, device=data.device, requires_grad=False)
	#all_labels = all_data.view(all_data.shape)[:, :num_sim_in_columns]
	all_input  = all_data.view(all_data.shape)[:, num_sim_in_columns:num_sim_in_out_columns]

	## Pass the numpy array through the model.
	with torch.no_grad():
		all_output = model(all_input)
	npoutput=all_output.numpy()

	## Reconstruct the Pandas frame with appropriate columns.
	input_columns = simulation_data.data.columns.values.tolist()

	predicted_columns = ["pred_{0:s}".format(name) for name in input_columns[:num_sim_in_columns]]

	output_columns = input_columns[:]
	output_columns[num_sim_in_out_columns:num_sim_in_out_columns] = predicted_columns[:]

	## Construct a new npoutput with the 7 new prediction columns added.
	npdata_extra = simulation_data.data.values
	expanded_npoutput = np.concatenate(
		(
			npdata_extra[:, :num_sim_in_out_columns],
			npoutput,
			npdata_extra[:, num_sim_in_out_columns:],
		),
		axis=1,
	)

	if use_gan:
		# If the input columns lacked GAN columns, then add them now, since the
		# GAN columns are present.
		if len(input_columns) <= num_sim_in_columns:
			# No GAN columns.  Add them.
			output_columns += ["GAN_{0:d}".format(gan_column_num) for gan_column_num in range(gan_n)]

	output = pd.DataFrame(
		data=expanded_npoutput,
		columns=output_columns,
	)

	# Check boundaries.
	input_npmins = np.array(simulation_data.simulation_info.sim_input_mins)
	input_npmaxs = np.array(simulation_data.simulation_info.sim_input_maxs)
	if not output_keep_out_of_bounds:
		# Get a mask of np.array([True, True, True, False, True, ...]) as to which rows are
		# valid.
		input_npmins_repeated = np.repeat(np.array([input_npmins]), npoutput.shape[0], axis=0)
		input_npmaxs_repeated = np.repeat(np.array([input_npmaxs]), npoutput.shape[0], axis=0)
		min_valid_npoutput = npoutput >= input_npmins_repeated
		max_valid_npoutput = npoutput <= input_npmaxs_repeated
		valid_npoutput = np.logical_and(min_valid_npoutput, max_valid_npoutput)
		#valid_npoutput_samples = np.apply_along_axis(all, axis=1, arr=valid_npoutput)[:,np.newaxis]  # Reduce rows by "and".
		valid_npoutput_mask = np.apply_along_axis(all, axis=1, arr=valid_npoutput)  # Reduce rows by "and" and get a flat, 1-D vector.

		# Only keep valid rows in output.
		old_num_samples = len(output)
		output = output.iloc[valid_npoutput_mask]
		new_num_samples = len(output)
		num_lost_samples = old_num_samples - new_num_samples

		if num_lost_samples <= 0:
			print("All model predictions are within the minimum and maximum boundaries.")
			print("")
		else:
			print("WARNING: #{0:d}/#{0:d} sample rows have been discarded from the CSV output due to out-of-bounds predictions.".format(num_lost_samples, old_num_samples))
			print("")

	# Make sure the output isn't all the same.
	if len(npoutput) >= 2:
		npoutput_means = np.apply_along_axis(np.std, axis=0, arr=npoutput)
		npoutput_stds = np.apply_along_axis(np.std, axis=0, arr=npoutput)
		# Warn if the std is <= this * (max_bound - min_bound).
		std_warn_threshold = 0.1
		num_warnings = 0
		unique_warn_threshold = 25
		for idx, name in enumerate(simulation_data.data.columns.values[:simulation_data.simulation_info.num_sim_inputs]):
			std = npoutput_stds[idx]
			this_threshold = std_warn_threshold * (input_npmaxs[idx] - input_npmins[idx])
			if std <= 0.0:
				print("WARNING: all predictions for simulation input parameter #{0:d} (`{1:s}`) are the same!  Prediction: {2:f}.".format(idx + 1, name, npoutput[0][idx]))
				num_warnings += 1
			elif std <= this_threshold:
				print("WARNING: there is little variance in the predictions for simulation input parameter #{0:d} (`{1:s}`): std <= this_threshold: {2:f} <= {3:f}.".format(idx + 1, name, std, this_threshold))
				num_warnings += 1
			else:
				# This may be inefficient, but count unique values and warn if
				# there are few.
				col = npoutput[:,idx]
				unique = set(npoutput[:,idx].tolist())
				if len(unique) <= unique_warn_threshold:
					print("WARNING: there are few unique values (#{0:d}) for predictions for simulation input parameter #{1:d} (`{2:s}`):".format(
						len(unique), idx + 1, name
					))
					for val in sorted(list(unique)):
						count = len([x for x in col if math.isclose(x, val)])
						if count > 1:
							print("  {0:s} (x{1:d})".format(str(val), count))
						else:
							print("  {0:s}".format(str(val)))
		if num_warnings >= 1:
			print("")

	# Write the output.
	simulation_data.save(output)
	print("Wrote CSV output with predictions to `{0:s}'.".format(save_data_path))

def stats():
	"""
	(To be documented...)
	"""
	print("(To be implemented...)")
	raise NotImplementedError("error: stats: the stats action is not yet implemented.")
	pass
