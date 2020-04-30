# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
Module providing some constants.
"""

import torch

# standardize is checked, then normalize_population, then normalize_bounds.

# Whether to standardize data input and read from a network (use mean and
# stddev).
standardize = True

# Whether to scale and translate data input and read from a model (use min and
# max) according to the distribution of the entire dataset used.
normalize_population = False

# Whether to scale and translate data input and read from a model (use min and
# max) according .
normalize_bounds = False

# When using either normalization technique, let the values be in the range
# [-1, -1] rather than [0, 1].
normalize_negative = True

def is_standardized():
	"""Determine whether any standardization or normalization is enabled."""
	return standardize or normalize_population or normalize_bounds

def is_standardized_negative():
	"""
	Determine whether any standardization or normalization is enabled and
	the standardized values can be negative.
	"""
	return is_standardized() and (standardize or normalize_negative)

# The default --gan-n value: the default number of additional input neurons to
# the generator beyond the 5 desired simulation output values.
default_gan_n = 8

# The default --num-epochs value: How many times to train this model over the
# entire dataset.
default_num_epochs = 100

# The default --status-every-epoch=n value: Output status every n epochs; 0 to
# disable.
default_status_every_epoch = 10

# The default --status-every-record=n value: Within an epoch whose status is
# displayed, output status every n records; 0 to disable.
default_status_every_sample = 1000

# The default --batch-size value.
# FIXME: the NN doesn't appear to be used at all for 0?
default_batch_size = 64

# 'cuda' if a CUDA-enabled GPU is available, otherwise 'cpu'.
# (From Braysen's example.py.)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# When training, how many records should we reserve for testing?
test_proportion = 0.25

# What learning rate should the optimizer use?
default_learning_rate = 0.02

# SGD parameters.

# momentum
momentum = 0.01
# lambda (L2 penalty)
weight_decay = 0.01
dampening = 0.01
nesterov = True
if nesterov:
	dampening = 0

# testing split seed.
testing_split_seed = 42

# Enable methods that try to automatically size certain groups of text output.
auto_size_formatting = True

# By default, can we pause training at most one GAN subnetwork if one is more
# accurate than the other?
default_gan_enable_pause = True

# GAN parameter: keep the generator and the discriminator loss in balance.  If
# the loss of the other is more than this value, pause training this one.
default_gan_training_pause_threshold = 0.3

# GAN parameter: don't pause training of a subnetwork if fewer than this many
# samples have been trained in an epoch.
#
# Set no 0 to disable the effect of this parameter.
default_pause_min_samples_per_epoch = 1024

# GAN parameter: don't pause training of a subnetwork if fewer than this many
# epochs have been run.
#
# Set no 0 to disable the effect of this parameter.
default_pause_min_epochs = 0

# GAN parameter: don't pause training of a subnetwork if fewer than this many
# epochs have been run.
#
# Set no 0 to disable the effect of this parameter.
default_pause_max_epochs = 0

# When training the discriminator with a reversed model, only correct the
# discriminator for generated input when it is more confident the generated
# input is correct than we are.
no_underconfident_discriminator = True
