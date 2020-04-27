#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
The CLI interface, providing main().
"""

import argparse
import logging
import logging.handlers
import os.path
import shlex
import sys
import textwrap

if True:
	# Let wcmi/cli.py be callable in a standalone directory.
	import sys, os.path
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wcmi.exception import WCMIArgsError
from wcmi.log import logger

import wcmi.log
import wcmi.nn as wnn
import wcmi.nn.data as data
import wcmi.nn.interface
import wcmi.nn.gan as gan
import wcmi.version

# Set default_action to None to require an action when none is provided.
default_action = "default"

def main(argv=None, logger=logger):
	"""
	Train or run the GAN.
	"""

	if argv is None:
		argv = sys.argv[:]

	args = argv[1:]

	try:
		return cli(args=args, argv=argv, logger=logger)
	except WCMIArgsError as ex:
		if False:
			parser = argument_parser
			parser.print_usage()
			if argv is not None and len(argv) >= 1:
				prog = argv[0]
				formatted_prog = os.path.basename(prog)
				logger.error("{0:s}: {1:s}".format(formatted_prog, str(ex)))
			else:
				logger.error(str(ex))
			sys.exit(2)
		else:
			parser = argument_parser
			err_msg = str(ex)
			remove_prefixes = (
				"error: ",
			)
			for remove_prefix in remove_prefixes:
				if err_msg.startswith(remove_prefix):
					err_msg = err_msg[len(remove_prefix):]
			parser.error(err_msg)

def cli(args=None, argv=None, parser=None, logger=logger):
	"""
	Process command-line arguments and respond accordingly.

	Optionally process command-line arguments without the program name instead
	of the default by setting `args`.  `args` is prefered to `argv`, which
	includes the program name.

	Note: the argument is `args', not `argv`; it does not include `prog` (e.g.
	`$0` in a shell, the program name.)
	"""
	# Determine the command-line arguments.
	if argv is None:
		argv = sys.argv[:]
	if args is None:
		args = argv[1:]
	if parser is None:
		parser = argument_parser

	# Perform preliminary argument parsing without a required action to check
	# for flags that don't require one, like --version.
	check_options, unrecognized = parser.parse_known_args(args + [""])

	# Handle --version.
	if check_options.version:
		print(wcmi.version.version_str)
		return

	# Parse arguments.
	options = parser.parse_args(args)

	# Handle --log, --log-truncate, --verbose, and --no-color.
	if not options.no_color:
		for handler in [wcmi.log.stdout_handler, wcmi.log.stderr_handler]:
			if handler is not None:
				handler.setFormatter(wcmi.log.console_formatter)

	if options.verbose:
		wcmi.log.verbose(logger=logger)

	if options.log is None:
		if options.log_truncate:
			raise WCMIArgsError("error: --log-truncate requires --log.")
		if options.log_timestamp:
			raise WCMIArgsError("error: --log-timestamp requires --log.")

	if options.log is not None:
		log_handler_kwargs = {
			**dict(
				mode="a" if not options.log_truncate else "w",
			),
		}
		log_handler = logging.handlers.WatchedFileHandler(
			options.log,
			**log_handler_kwargs,
		)
		log_handler.setFormatter(wcmi.log.default_formatter if not options.log_timestamp else wcmi.log.timestamped_formatter)
		logger.addHandler(log_handler)

	logger.debug("Logging enabled.")
	logger.debug(" ".join(shlex.quote(arg) for arg in argv))

	# Handle and go to the action.
	if "action" not in options:
		raise WCMIArgsError("error: no action specified.  Try passing train, run, or stats.")
	action = options.action
	if action not in actions:
		raise WCMIArgsError("error: unrecognized action `{0:s}'.  Try passing train, run, or stats.".format(action))
	return actions[action](options, parser=parser, logger=logger)

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

			Actions:
			  train --(gan|dense) [--load-model=PATH] --load-data=PATH --save-model=PATH [--save-data=PATH]
			    Load input CSV data, train the model, and save its parameters
			    (weights and biases.)

			    Optionally, perform additional training on a pretrained model
			    providing a path to an already saved model.

			    Optionally, save epoch loss CSV data (not output data) to the
			    --save-data path.

			    Note: for training, --save-data outputs MSE accuracy data, not
			    predictions output!  For predictions output, use "run", not
			    "train".

			  run --(gan|dense) --load-model=PATH --load-data=PATH --save-data=PATH
			    Load input CSV data and a model, add prediction columns to the data,
			    and write a new CSV file with the prediction columns added.

			  stats --load-data=PATH --save-data=DIR_PATH
			    Load either *output* CSV data (with added prediction columns)
			    or epoch MSE CSV data and output various stats files somewhere
			    inside the directory at DIR_PATH.

			  default
			    Run a typical sequence of train, run, and stats commands.

			CSV formats:
			  There are 3 recognized CSV formats: input data (12 + n columns),
			  prediction data (17 + n columns, with 7 additional model output
			  prediction columns; input plus predictions), and (training/epoch)
			  loss data.

			  Here are more details about these CSV formats:

			  Input CSV data (train --load-data or run --load-data): 12 + n>=0 total columns:
			    7 simulation inputs, then 5 simulation outputs, then optionally forced additional GAN parameters:
			      Iin[A],Iout[A],l[mm],p1[mm],p2[mm],p3[mm],win[mm],kdiff[%],Bleak[uT],V_PriWind[cm3],V_PriCore[cm3],Pout[W]

			    If the additional --gan-n columns are absent, random values will
			    be chosen.

			    The column names must be recognized.  The n GAN column names must
			    begin with "GAN", e.g. "GAN_0", "GAN_1", "GAN_2", etc., or
			    "GAN_brightness".

			  Prediction CSV (run --save-data or stats --load-data): 19 + n>=0 total columns:
			    7 simulation inputs, 5 simulation outputs, 7 predicted simulation inputs (model outputs), n>=0 generator parameters (if GAN).

			    The 7 predicted simulation inputs that are output by the model
			    have a column name prefixed with "pred_".

			  Loss CSV (train --save-data or stats --load-data): variable column count:
			    The column count depends on the model.  See documentation or
			    comments and code in wcmi/nn/interface.py for more information.

			Examples:
			  Train new model dense.pt:
			    ./wcmi.py train --dense --load-data data/4th_dataset_noid.csv --save-model dist/dense.pt
			  Run model dense.pt and output to a new CSV file:
			    ./wcmi.py run   --dense --load-model dist/dense.pt --load-data data/4th_dataset_noid.csv --save-data dist/4th_dataset_dense_predictions.csv
			  Perform additional training on model dense.pt:
			    ./wcmi.py train --dense --load-model dist/dense.pt --save-model dist/dense.pt --load-data data/4th_dataset_noid.csv

			More complete examples:
			  Train new model dense.pt:
			    ./wcmi.py train --dense --load-data=data/4th_dataset_noid.csv --save-model=dist/dense.pt --save-data=dist/4th_dataset_dense_mse.csv --batch-size=64 --num-epochs=100 --log=dist/log/train_dense_00_initial.log --log-truncate
			  Perform additional training on dense.pt:
			    ./wcmi.py train --dense --load-model=dist/dense.pt --load-data=data/4th_dataset_noid.csv --save-model=dist/dense.pt --save-data=dist/4th_dataset_dense_mse.csv --batch-size=64 --num-epochs=100 --log=./dist/log/train_dense_01_repeat.log --log-truncate
			  Run model dense.pt and output to a new CSV file:
			    ./wcmi.py run --dense --load-model=dist/dense.pt --load-data=data/4th_dataset_noid.csv --save-data=dist/4th_dataset_dense_predictions.csv --log=dist/log/run_dense.log --log-truncate

			Notes:
			  (To rearrange the columns of an ndarray with a numpy permutation,
			  this may be a helpful post: https://stackoverflow.com/a/20265477)
		"""),
	}
	parser = argparse.ArgumentParser(**argparse_kwargs)

	parser.add_argument("-V", "--version", action="store_true", help="(All actions): print the current version.")
	parser.add_argument("-v", "--verbose", action="store_true", help="(All actions): increase verbosity: print more messages to standard output.")
	parser.add_argument("--no-color", action="store_true", help="(All actions): disable coloring console output.")

	parser.add_argument("--log", type=str, help="(All actions): copy all output to the given log file, by default appending to it.")
	parser.add_argument("--log-truncate", action="store_true", help="(All actions): overwrite existing log files.")
	parser.add_argument("--log-timestamp", action="store_true", help="(All actions): include timestamps in log files.")

	if default_action is None:
		parser.add_argument("action", type=str, help="Specify what to do: train, run, or stats.")
	else:
		parser.add_argument("action", type=str, nargs="?", default=default_action, help="Specify what to do: train, run, or stats.")

	parser.add_argument("--dense", action="store_true", help="(All actions): use the dense model (ANN) rather than the GAN.")
	parser.add_argument("--gan", action="store_true", help="(All actions): use the GAN rather than the dense model.")
	parser.add_argument("--load-model", type=str, help="(All actions): load a pretrained model rather than randomly initialize the model chosen.")

	parser.add_argument("--save-model", type=str, help="(All actions): after training, save the model to this file.")
	parser.add_argument(
		"--gan-n", "--gan-num-gen-params", type=int, default=gan.default_gan_n,
		help="(train action): if using the GAN, specify the number of additional GAN generator parameters (default: {0:d}).  Fail when loading CSV file with a different --gan-n setting.".format(
			gan.default_gan_n
		),
	)

	parser.add_argument("--load-data", type=str, help="(All actions): load training data from this CSV file.")

	parser.add_argument("--save-data", type=str, help="(All actions): after running the neural network model on the loaded CSV data, output ")

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

	parser.add_argument(
		"--learning-rate", type=int, default=data.default_learning_rate,
		help="(train action): specify the learning rate that the optimizer should use (default: {0:f}).".format(
			data.default_learning_rate,
		),
	)

	parser.add_argument(
		"--gan-force-fixed-gen-params",
		action="store_true",
		help="(train --gan action): When training GAN data, instead of using random generation parameters, use the fixed parameters provided by the input CSV data."
	)

	parser.add_argument("--output-keep-out-of-bounds-samples", action="store_true", help="(run action): For CSV predictions output only, keep rows with out-of-bounds predictions.")

	# GAN training parameters.

	parser.add_argument(
		"--gan-disable-pause", action="store_false", dest="gan_enable_pause",
		help="(run --gan action): disable pausing training a subnetwork under certain conditions.",
	)
	parser.add_argument(
		"--gan-training-pause-threshold", type=float, default=data.default_gan_training_pause_threshold,
		help="(run --gan action): if a subnetwork outperforms the other by this quantity, pause training it.",
	)
	parser.add_argument(
		"--pause-min-samples-per-epoch", type=int, default=data.default_pause_min_samples_per_epoch,
		help="(run --gan action): don't pause if fewer than this many samples have been trained on in the batches of this epoch.",
	)
	parser.add_argument(
		"--pause-min-epochs", type=int, default=data.default_pause_min_epochs,
		help="(run --gan action): don't pause if fewer than this many epochs have been trained.",
	)
	parser.add_argument(
		"--pause-max-epochs", type=int, default=data.default_pause_max_epochs,
		help="(run --gan action): don't pause if more than this many epochs have been trained (set to 0 to disable).",
	)

	return parser

argument_parser = get_argument_parser()

actions = {}
def add_action(func):
	actions[func.__name__] = func
	return func

def verify_model_options(options):
	"""
	Perform CLI argument verification common actions that use a model.
	"""

	# Check --gan and --dense.
	if options.dense and options.gan:
		raise WCMIArgsError("error: both --gan and --dense were specified.")
	if not options.dense and not options.gan:
		raise WCMIArgsError("error: please pass either --gan to use the GAN or --dense to use the dense model.")
	if not (options.dense != options.gan):
		# (This is redundant.)
		raise WCMIArgsError("error: --gan or --dense must be specified, but not both.")

def verify_common_options(options):
	"""
	Perform CLI argument verification common to all actions.
	"""

	# Ensure multiple models are not specified at the same time.
	if options.dense and options.gan:
		raise WCMIArgsError("error: both --gan and --dense were specified.")

	# Check --gan-n.
	if options.gan_n < 0:
		raise WCMIArgsError("error: --gan-n must be provided with a non-negative number, but {0:d} was provided.".format(options.gan_n))

	# Check --gan-force-fixed-gen-params is specified with --gan.
	if options.gan_force_fixed_gen_params and not options.gan:
		raise WCMIArgsError("error: --gan-force-fixed-gen-params requires --gan.")

def verify_model_options(options):
	"""
	Perform CLI argument verification common actions that use a model.
	"""

	# Check --gan and --dense.

	# Ensure at least one model is specified.
	if not options.dense and not options.gan:
		raise WCMIArgsError("error: please pass either --gan to use the GAN or --dense to use the dense model.")

	# Ensure --dense iff not --gan.
	if not (options.dense != options.gan):
		# (This is redundant.)
		raise WCMIArgsError("error: --gan or --dense must be specified, but not both.")

def verify_load_data_options(options):
	"""
	Perform CLI argument verification common to actions that load data.
	"""

	# Check --load-data.
	if options.load_data is None:
		raise WCMIArgsError("error: --load-data .../path/to/data.csv must be specified.")

@add_action
def train(options, parser=argument_parser, logger=logger):
	"""
	Call the train action after some argument verification.
	"""

	# Verify command-line arguments.
	verify_common_options(options)
	verify_model_options(options)
	verify_load_data_options(options)

	if options.save_model is None:
		raise WCMIArgsError("error: the train action requires --save-model.")

	if options.output_keep_out_of_bounds_samples:
		raise WCMIArgsError("error: the train action doesn't support --output-keep-out-of-bounds-samples.")

	# Call the action.
	return wnn.interface.train(
		use_gan=options.gan,
		load_model_path=options.load_model,
		save_model_path=options.save_model,
		load_data_path=options.load_data,
		save_data_path=options.save_data,
		gan_n=options.gan_n,
		num_epochs=options.num_epochs,
		status_every_epoch=options.status_every_epoch,
		status_every_sample=options.status_every_sample,
		batch_size=options.batch_size,
		learning_rate=options.learning_rate,
		gan_force_fixed_gen_params=options.gan_force_fixed_gen_params,
		gan_enable_pause=options.gan_enable_pause,
		gan_training_pause_threshold=options.gan_training_pause_threshold,
		pause_min_samples_per_epoch=options.pause_min_samples_per_epoch,
		pause_min_epochs=options.pause_min_epochs,
		pause_max_epochs=options.pause_max_epochs,
		logger=logger,
	)

@add_action
def run(options, parser=argument_parser, logger=logger):
	"""
	Call the run action after some argument verification.
	"""

	# Verify command-line arguments.
	verify_common_options(options)
	verify_model_options(options)
	verify_load_data_options(options)

	if options.load_model is None:
		raise WCMIArgsError("error: the run action requires --load-model.")

	if options.save_model is not None:
		raise WCMIArgsError("error: the run action doesn't support --save-model.")

	if options.gan_force_fixed_gen_params:
		raise WCMIArgsError(
			"error: the run action doesn't support --gan-force-fixed-gen-params.  run will use fixed GAN generation parameters instead of random noise only if the input CSV data specified them.",
		)

	if options.batch_size != data.default_batch_size:
		raise WCMIArgsError("error: the run action doesn't support --batch-size.")
	if options.learning_rate != data.default_learning_rate:
		raise WCMIArgsError("error: the run action doesn't support --learning-rate.")
	if options.gan_enable_pause != data.default_gan_enable_pause:
		raise WCMIArgsError("error: the run action doesn't support --gan-disable-pause.")
	if options.gan_training_pause_threshold != data.default_gan_training_pause_threshold:
		raise WCMIArgsError("error: the run action doesn't support --gan-training-pause-threshold.")
	if options.pause_min_samples_per_epoch != data.default_pause_min_samples_per_epoch:
		raise WCMIArgsError("error: the run action doesn't support --pause-min-samples-per-epoch.")
	if options.pause_min_epochs != data.default_pause_min_epochs:
		raise WCMIArgsError("error: the run action doesn't support --pause-min-epochs.")
	if options.pause_max_epochs != data.default_pause_max_epochs:
		raise WCMIArgsError("error: the run action doesn't support --pause-max-epochs.")

	# Call the action.
	return wnn.interface.run(
		use_gan=options.gan,
		load_model_path=options.load_model,
		load_data_path=options.load_data,
		save_data_path=options.save_data,
		gan_n=options.gan_n,
		output_keep_out_of_bounds_samples=options.output_keep_out_of_bounds_samples,
		logger=logger,
	)

@add_action
def stats(options, parser=argument_parser, logger=logger):
	"""
	Call the run action after some argument verification.
	"""

	# Verify command-line arguments.
	verify_common_options(options)
	verify_load_data_options(options)

	if options.load_data is None:
		raise WCMIArgsError("error: the stats action requires --load-data.")
	if options.save_data is None:
		raise WCMIArgsError("error: the stats action requires --save-data.")
	if options.load_model is None:
		raise WCMIArgsError("error: the stats action doesn't support --load-model.")
	if options.save_model is None:
		raise WCMIArgsError("error: the stats action doesn't support --save-model.")

	if options.output_keep_out_of_bounds_samples:
		raise WCMIArgsError("error: the stats action doesn't support --output-keep-out-of-bounds-samples.")
	if options.gan_force_fixed_gen_params:
		raise WCMIArgsError("error: the stats action doesn't support --gan-force-fixed-gen-params.")

	if options.batch_size != data.default_batch_size:
		raise WCMIArgsError("error: the stats action doesn't support --batch-size.")
	if options.learning_rate != data.default_learning_rate:
		raise WCMIArgsError("error: the stats action doesn't support --learning-rate.")
	if options.gan_enable_pause != data.default_gan_enable_pause:
		raise WCMIArgsError("error: the stats action doesn't support --gan-disable-pause.")
	if options.gan_training_pause_threshold != data.default_gan_training_pause_threshold:
		raise WCMIArgsError("error: the stats action doesn't support --gan-training-pause-threshold.")
	if options.pause_min_samples_per_epoch != data.default_pause_min_samples_per_epoch:
		raise WCMIArgsError("error: the stats action doesn't support --pause-min-samples-per-epoch.")
	if options.pause_min_epochs != data.default_pause_min_epochs:
		raise WCMIArgsError("error: the stats action doesn't support --pause-min-epochs.")
	if options.pause_max_epochs != data.default_pause_max_epochs:
		raise WCMIArgsError("error: the stats action doesn't support --pause-max-epochs.")

	# Call the action.
	return wnn.interface.stats(
		logger=logger,
	)

def get_default_actions(parser=argument_parser):
	"""
	Get a tuple of actions and options to pass to them to run for the
	"default" action.
	"""
	return (
		("train", {
			"dense":        (True,                                  "--dense"),
			"load_data":    ("data/4th_dataset_noid.csv",           "--load-data=data/4th_dataset_noid.csv"),
			"save_model":   ("dist/dense_00_initial.pt",            "--save-model=dist/dense_00_initial.pt"),
			"save_data":    ("dist/train_dense_mse_00_initial.csv", "--save-data=dist/train_dense_mse_00_initial.csv"),
			"log":          ("dist/log/train_dense_00_initial.log", "--log=dist/log/train_dense_00_initial.log"),
			"log_truncate": (True,                                  "--log-truncate"),
		}),
		("train", {
			"dense":        (True,                                  "--dense"),
			"load_model":   ("dist/dense_00_initial.pt",            "--load-model=dist/dense_00_initial.pt"),
			"load_data":    ("data/4th_dataset_noid.csv",           "--load-data=data/4th_dataset_noid.csv"),
			"save_model":   ("dist/dense.pt",                       "--save-model=dist/dense.pt"),
			"save_data":    ("dist/train_dense_mse_01_repeat.csv",  "--save-data=dist/train_dense_mse_01_repeat.csv"),
			"log":          ("dist/log/train_dense_01_repeat.log",  "--log=dist/log/train_dense_01_repeat.log"),
			"log_truncate": (True,                                  "--log-truncate"),
		}),
		("train", {
			"gan":          (True,                                  "--gan"),
			"load_data":    ("data/4th_dataset_noid.csv",           "--load-data=data/4th_dataset_noid.csv"),
			"save_model":   ("dist/gan_00_initial.pt",              "--save-model=dist/gan_00_initial.pt"),
			"save_data":    ("dist/train_gan_bce_00_initial.csv",   "--save-data=dist/train_gan_bce_00_initial.csv"),
			#"pause_min_samples_per_epoch": (1024,                   "--pause-min-samples-per-epoch=1024"),
			"log":          ("dist/log/train_gan_00_initial.log",   "--log=dist/log/train_gan_00_initial.log"),
			"log_truncate": (True,                                  "--log-truncate"),
		}),
		("train", {
			"gan":          (True,                                  "--gan"),
			"load_model":   ("dist/gan_00_initial.pt",              "--load-model=dist/gan_00_initial.pt"),
			"load_data":    ("data/4th_dataset_noid.csv",           "--load-data=data/4th_dataset_noid.csv"),
			"save_model":   ("dist/gan.pt",                         "--save-model=dist/gan.pt"),
			"save_data":    ("dist/train_gan_bce_01_repeat.csv",    "--save-data=dist/train_gan_bce_01_repeat.csv"),
			"log":          ("dist/log/train_gan_01_repeat.log",    "--log=dist/log/train_gan_01_repeat.log"),
			"log_truncate": (True,                                  "--log-truncate"),
		}),

		("run", {
			"dense":        (True,                                  "--dense"),
			"load_model":   ("dist/dense.pt",                       "--load-model=dist/dense.pt"),
			"load_data":    ("data/4th_dataset_noid.csv",           "--load-data=data/4th_dataset_noid.csv"),
			"save_data":    ("dist/4th_dataset_dense_predictions.csv", "--save-data=dist/4th_dataset_dense_predictions.csv"),
			"log":          ("dist/log/run_dense.log",              "--log=dist/log/run_dense.log"),
			"log_truncate": (True,                                  "--log-truncate"),
		}),
		("run", {
			"gan":          (True,                                  "--gan"),
			"load_model":   ("dist/gan.pt",                         "--load-model=dist/gan.pt"),
			"load_data":    ("data/4th_dataset_noid.csv",           "--load-data=data/4th_dataset_noid.csv"),
			"save_data":    ("dist/4th_dataset_gan_predictions.csv", "--save-data=dist/4th_dataset_gan_predictions.csv"),
			"log":          ("dist/log/run_gan.log",                "--log=dist/log/run_gan.log"),
			"log_truncate": (True,                                  "--log-truncate"),
		}),

		("stats", {
			"load_data":    ("dist/4th_dataset_dense_mse.csv",      "--load-data=dist/4th_dataset_dense_mse.csv"),
			"save_data":    ("dist/stats",                          "--save-data=dist/stats"),
			"log":          ("dist/log/stats_dense_mse.log",        "--log=dist/log/stats_dense_mse.log"),
			"log_truncate": (True,                                  "--log-truncate"),
		}),
		("stats", {
			"load_data":    ("dist/4th_dataset_gan_bce.csv",        "--load-data=dist/4th_dataset_gan_bce.csv"),
			"save_data":    ("dist/stats",                          "--save-data=dist/stats"),
			"log":          ("dist/log/stats_gan_bce.log",          "--log=dist/log/stats_gan_bce.log"),
			"log_truncate": (True,                                  "--log-truncate"),
		}),
		("stats", {
			"load_data":    ("dist/4th_dataset_dense_predictions.csv", "--load-data=dist/4th_dataset_dense_predictions.csv"),
			"save_data":    ("dist/stats",                          "--save-data=dist/stats"),
			"log":          ("dist/log/stats_dense_predictions.log", "--log=dist/log/stats_dense_predictions.log"),
			"log_truncate": (True,                                  "--log-truncate"),
		}),
		("stats", {
			"load_data":    ("dist/4th_dataset_gan_predictions.csv", "--load-data=dist/4th_dataset_gan_predictions.csv"),
			"save_data":    ("dist/stats",                          "--save-data=dist/stats"),
			"log":          ("dist/log/stats_gan_predictions.log",  "--log=dist/log/stats_gan_predictions.log"),
			"log_truncate": (True,                                  "--log-truncate"),
		}),
	)

default_actions = get_default_actions()

@add_action
def default(options, parser=argument_parser, default_actions=default_actions, logger=logger):
	"""
	Run a typical sequence of train, run, and stats commands.
	"""

	# Verify command-line arguments.
	verify_common_options(options)

	# Ensure "default" isn't called with any options that we override.
	for action, action_options in default_actions:
		for option_key, (option_value, flag) in action_options.items():
			if option_key in options:
				if getattr(options, option_key) != parser.get_default(option_key):
					raise WCMIArgsError("error: the `default' action provides the flag `{0:s}', but a flag for the target option was passed.".format(flag))

	# Get non-default options to call each action with in addition to the
	# action-specific ones.
	options_dict = dict()
	try:
		options_dict = {**options.__dict__}
	except (TypeError, ValueError, AttributeError, KeyError) as ex:
		raise ex

	# Prepare to run each action.
	is_python_3_8 = True
	try:
		shlex_join = shlex.join
		is_python_3_8 = True
	except AttributeError as ex:
		is_python_3_8 = False
	if is_python_3_8:
		def format_action(action, action_options):
			"""
			Return a string showing the command to run without prog, e.g.
			"train --dense --load-data=data/4th_dataset_noid.csv --save-model=dist/dense.pt --save-data=dist/4th_dataset_dense_mse.csv"
			or
			"run --dense --load-model=dist/dense.pt --load-data=data/4th_dataset_noid.csv --save-data=dist/4th_dataset_dense_predictions.csv"
			or
			"train --load-data=dist/4th_dataset_dense_mse.csv --save-data=dist/stats"
			or
			"train --load-data=dist/4th_dataset_dense_predictions.csv --save-data=dist/stats"
			"""
			return "{0:s} {1:s}".format(action, shlex.join(flag for option_key, (option_value, flag) in action_options.items()))
	else:
		def format_action(action, action_options):
			"""
			Return a string showing the command to run without prog, e.g.
			"train --dense --load-data=data/4th_dataset_noid.csv --save-model=dist/dense.pt --save-data=dist/4th_dataset_dense_mse.csv"
			or
			"run --dense --load-model=dist/dense.pt --load-data=data/4th_dataset_noid.csv --save-data=dist/4th_dataset_dense_predictions.csv"
			or
			"train --load-data=dist/4th_dataset_dense_mse.csv --save-data=dist/stats"
			or
			"train --load-data=dist/4th_dataset_dense_predictions.csv --save-data=dist/stats"
			"""
			return "{0:s} {1:s}".format(action, " ".join(shlex.quote(flag) for option_key, (option_value, flag) in action_options.items()))

	def run_action(action, action_options, logger=logger):
		"""Run an individual action with options."""
		return actions[action](
			argparse.Namespace(**{
				**options_dict,
				**{option_key: option_value for option_key, (option_value, flag) in action_options.items()},
			}),
			logger=logger,
		)
	def run_actions(actions=default_actions, run_action=run_action, logger=logger):
		"""Print the action to run and run it for each action."""
		last_result = None
		for action, action_options in actions:
			logger.info(format_action(action, action_options))
			last_result = run_action(action, action_options, logger=logger)
		return last_result

	# Now run each action.
	logger.info("Will run the following actions:")
	for action, action_options in default_actions:
		logger.info("  {0:s}".format(format_action(action, action_options)))
	logger.info("")

	raise NotImplementedError("FIXME: the `default' action currently doesn't write log files; disabling the `default' action until this bug is fixed.")
	run_actions(default_actions, logger=logger)

	logger.info("")
	logger.info("Done running default actions.")

if __name__ == "__main__":
	import sys
	main(sys.argv[:])
