# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
Module providing some constants.
"""

import torch

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
default_batch_size = 64

# 'cuda' if a CUDA-enabled GPU is available, otherwise 'cpu'.
# (From Braysen's example.py.)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# When training, how many records should we reserve for testing?
test_proportion = 0.25

# What learning rate should the optimizer use?
learning_rate = 0.02

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
