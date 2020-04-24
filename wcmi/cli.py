#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
The CLI interface, providing main().
"""

import argparse
import os.path
import sys
import textwrap

if True:
	# Let wcmi/cli.py be callable in a standalone directory.
	import sys, os.path
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wcmi.exception import WCMIArgsError

import wcmi.nn as wnn
import wcmi.nn.data as data
import wcmi.nn.interface
import wcmi.nn.gan as gan
import wcmi.version

def main(argv=None):
	"""
	Train or run the GAN.
	"""

	if argv is None:
		argv = sys.argv[:]

	args = argv[1:]

	try:
		return cli(args=args)
	except WCMIArgsError as ex:
		parser = argument_parser
		parser.print_usage()
		if argv is not None and len(argv) >= 1:
			prog = argv[0]
			formatted_prog = os.path.basename(prog)
			print("{0:s}: {1:s}".format(formatted_prog, str(ex)))
		else:
			print(str(ex))
		return

def cli(args=None):
	"""
	Process command-line arguments and respond accordingly.

	Note: the argument is `args', not `argv`; it does not include `prog` (e.g.
	`$0` in a shell, the program name.)
	"""
	# Determine the command-line arguments.
	if args is None:
		args = sys.argv[1:]

	# Parse arguments.
	parser = argument_parser
	options = parser.parse_args(args)

	# Handle --version.
	if options.version:
		print(wcmi.version.version_str)
		return

	# Handle and go to the action.
	if "action" not in options:
		raise WCMIArgsError("error: no action specified.  Try passing train, run, or stats.")
	action = options.action
	if action not in actions:
		raise WCMIArgsError("error: unrecognized action `{0:s}'.  Try passing train, run, or stats.".format(action))
	return actions[action](options)

def get_argument_parser(prog=None):
	"""
	Get a copy of the CLI argument parser.
	"""
	argparse_kwargs = {}
	if prog is not None:
		argparse_kwargs['prog'] = prog
	argparse_kwargs = {**argparse_kwargs,
		'formatter_class': argparse.RawDescriptionHelpFormatter,

		'description': textwrap.dedent("""\
			Train or run the neural network.
		"""),

		'epilog': textwrap.dedent("""\
			Train or run the neural network.

			Examples:
			  Train new model dense.pt:
			    ./main.py train --gan --save-model dist/dense.pt --load-data data/4th_dataset_noid.csv
			  Run model dense.pt and output to a new CSV file:
			    ./main.py run   --gan --load-model dist/dense.pt --load-data data/4th_dataset_noid.csv --save-data dist/4th_dataset_dense_predictions.csv
			  Perform additional training on model dense.pt:
			    ./main.py train --gan --load-model dist/dense.pt --save-model dist/dense.pt --load-data data/4th_dataset_noid.csv

			Training (--load-data) CSV columns (12 + n>=0 total):
			  7 simulation inputs, then 5 simulation outputs, then optionally forced additional GAN parameters:
			    Iin[A],Iout[A],l[mm],p1[mm],p2[mm],p3[mm],win[mm],kdiff[%],Bleak[uT],V_PriWind[cm3],V_PriCore[cm3],Pout[W]

			  If the additional --gan-n columns are absent, random values will
			  be chosen.

			  The column names must be recognized.  The n GAN column names must
			  begin with "GAN", e.g. "GAN_0", "GAN_1", "GAN_2", etc., or
			  "GAN_brightness".

			Post-running (--save-data) CSV columns (19 + n>=0 total):
			  7 simulation inputs, 5 simulation outputs, 7 predicted simulation inputs (model outputs), n>=0 generator parameters (if GAN).

			  The 7 predicted simulation inputs that are output by the model
			  have a column name prefixed with "pred_".

			(To re-arrange the columns with a numpy permutation, this may be a
			helpful post: https://stackoverflow.com/a/20265477)
		"""),
	}
	parser = argparse.ArgumentParser(**argparse_kwargs)
	parser.add_argument("action", type=str, help="Specify what to do: train, run, or stats.")

	parser.add_argument("--dense", action="store_true", help="(All actions): use the dense model (ANN) rather than the GAN.")
	parser.add_argument("--gan", action="store_true", help="(All actions): use the GAN rather than the dense model.")
	parser.add_argument("--load-model", type=str, help="(All actions): load a pretrained model rather than randomly initialize the model chosen.")

	parser.add_argument("--save-model", type=str, help="(All actions): after training, save the model to this file.")
	parser.add_argument(
		"--gan-n", "--gan-n-parameters", type=int, default=gan.default_gan_n,
		help="(train action): if using the GAN, specify the number of additional GAN generator parameters (default: {0:d}).  Fail when loading CSV file with a different --gan-n setting.".format(
			gan.default_gan_n
		),
	)

	parser.add_argument("--load-data", type=str, help="(All actions): load training data from this CSV file.")

	parser.add_argument("--save-data", type=str, help="(run action): after running the neural network model on the loaded CSV data, output ")

	parser.add_argument(
		"--num-epochs", type=int, default=data.default_num_epochs,
		help="(train action): how many times to train this model over the entire dataset (default: {0:d}); 0 to disable.".format(
			data.default_num_epochs,
		),
	)

	parser.add_argument(
		"--status-every-epoch", type=int, default=data.default_status_every_epoch,
		help="(train action): output status every n epochs (default: {0:d}); 0 to disable.".format(
			data.default_status_every_epoch,
		),
	)

	parser.add_argument(
		"--status-every-sample", type=int, default=data.default_status_every_sample,
		help="(train action): within an epoch whose status is displayed, output status every n samples (default: {0:d}); 0 to disable.".format(
			data.default_status_every_sample,
		),
	)

	parser.add_argument(
		"--batch-size", type=int, default=data.default_batch_size,
		help="(train action): specify the batch size to use when training (default: {0:d}).".format(
			data.default_batch_size,
		),
	)

	parser.add_argument("-V", "--version", action="store_true", help="(All actions): Print the current version.")

	return parser

argument_parser = get_argument_parser()

actions = {}
def add_action(func):
	actions[func.__name__] = func
	return func

def verify_options_common(options):
	"""
	Perform CLI argument verification common to all actions.
	"""

	# Check --gan and --dense.
	if options.dense and options.gan:
		raise WCMIArgsError("error: both --gan and --dense were specified.")
	if not options.dense and not options.gan:
		raise WCMIArgsError("error: please pass either --gan to use the GAN or --dense to use the dense model.")
	if not (options.dense != options.gan):
		# (This is redundant.)
		raise WCMIArgsError("error: --gan or --dense must be specified, but not both.")

	# Check --gan-n.
	if options.gan_n < 0:
		raise WCMIArgsError("error: --gan-n must be provided with a non-negative number, but {0:d} was provided.".format(options.gan_n))

	# Check --load-data.
	if options.load_data is None:
		raise WCMIArgsError("error: --load-data .../path/to/data.csv must be specified.")

@add_action
def train(options):
	# Verify command-line arguments.
	verify_options_common(options)

	if options.save_model is None:
		raise WCMIArgsError("error: the train action requires --save-model.")

	if options.save_data is not None:
		raise WCMIArgsError("error: the train action doesn't support --save-data.")

	# Call the action.
	return wnn.interface.train(
		use_gan=options.gan,
		load_model_path=options.load_model,
		save_model_path=options.save_model,
		load_data_path=options.load_data,
		gan_n=options.gan_n,
		num_epochs=options.num_epochs,
		status_every_epoch=options.status_every_epoch,
		status_every_sample=options.status_every_sample,
	)

@add_action
def run(options):
	# Verify command-line arguments.
	verify_options_common(options)

	if options.load_model is None:
		raise WCMIArgsError("error: the run action requires --load-model.")

	if options.save_model is not None:
		raise WCMIArgsError("error: the run action doesn't support --save-model.")

	# Call the action.
	return wnn.interface.run(
		use_gan=options.gan,
		load_model_path=options.load_model,
		load_data_path=options.load_data,
		save_data_path=options.save_data,
		gan_n=options.gan_n,
	)

@add_action
def stats(options):
	# Verify command-line arguments.
	verify_options_common(options)

	# Call the action.
	return wnn.interface.stats()

if __name__ == "__main__":
	import sys
	main(sys.argv[:])
