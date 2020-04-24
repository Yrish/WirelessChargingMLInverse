# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
Module providing methods to interface with the neural networks provided by this
package.
"""

import numpy as np
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
	load_data_path=None, gan_n=gan.default_gan_n,
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
		save_data_path=None,
		verify_gan_n=True,
		optional_gan_n=True,
		gan_n=gan_n,
		simulation_info=simulation.simulation_info,
	)

	# Data verification.
	if len(simulation_data.data) <= 0:
		raise WCMIError("error: train requires at least one sample in the CSV file.")

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
		**mdl_kwargs,
	)
	# If CUDA is available, move the model to the GPU.
	model = model.to(data.device)

	# Train the model.
	if not use_gan:
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

		for fmt, tvec in (
			("", None),
			("Last testing MSE   (norm) : {0:s} ({1:f})", last_testing_mse),
			("Last testing RMSE  (norm) : {0:s} ({1:f})", last_testing_mse.sqrt()),
			("Last training MSE  (norm) : {0:s} ({1:f})", last_training_mse),
			("Last training RMSE (norm) : {0:s} ({1:f})", last_training_mse.sqrt()),
			("", None),
			(
				"Label column names: {0:s}".format(
					"<{0:s}>".format(", ".join(["{0:s}".format(column) for column in simulation_data.simulation_info.sim_input_names])),
				),
				None,
			),
			("", None),
			("All labels mean    (norm) : {0:s} ({1:f})", all_labels.mean(0)),
			("All labels var     (norm) : {0:s} ({1:f})", all_labels.var(0)),
			("All labels stddev  (norm) : {0:s} ({1:f})", all_labels.std(0)),
			("", None),
			("All labels min     (norm) : {0:s} ({1:f})", torch.Tensor(np.quantile(all_nplabels, 0, 0))),
			("...1st quartile    (norm) : {0:s} ({1:f})", torch.Tensor(np.quantile(all_nplabels, 0.25, 0))),
			("All labels median  (norm) : {0:s} ({1:f})", torch.Tensor(np.quantile(all_nplabels, 0.5, 0))),
			("...3rd quartile    (norm) : {0:s} ({1:f})", torch.Tensor(np.quantile(all_nplabels, 0.75, 0))),
			("All labels max     (norm) : {0:s} ({1:f})", torch.Tensor(np.quantile(all_nplabels, 1, 0))),
		):
			if tvec is None:
				print(fmt)
			else:
				print(
					fmt.format(
						"<{0:s}>".format(", ".join(["{0:f}".format(component) for component in tvec])),
						tvec.norm(),
					)
				)

		print("")
		print("Done training all epochs.")
		print("Have a good day.")

	else:
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
	num_sim_in_columns     = simulation_data.simulation_info.num_sim_inputs
	num_sim_in_out_columns = num_sim_in_columns + simulation_data.simulation_info.num_sim_outputs

	## Pass the numpy array through the model.
	npoutput = model(simulation_data.data.values[:,:num_sim_in_columns])

	## Reconstruct the Pandas frame with appropriate columns.
	input_columns = simulation_data.data.columns.values.tolist()

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
