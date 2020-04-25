#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
Logging utilities.
"""

from contextlib import AbstractContextManager

import copy
import logging
import sys

from wcmi.exception import WCMIError

logger_name  = __name__  # "wcmi.log"

class PushDefaultLoggingClassContext(AbstractContextManager):
	"""
	When used with `with', stores logging.getLoggerClass() and restores it with
	logging.setLoggerClass().

	Example:
		with PushDefaultLoggingClassContext():
			...
	"""

	propogate = False

	def __init__(self, logger_class=None, logger_name=None):
		"""
		Enable using `with' to store logging.getLoggerClass() and restore it
		afterwards.

		To set logging.getLoggerClass() in this context, pass in the new logger
		context as `logger_class`.

		To get a new logger with this class, also pass a new logger_name that
		doesn't refer to an existing logger.

		Example:
			with PushDefaultLoggingClassContext(MyLoggerClass, "my.logger.name") as logger:
				my_logger = logger
		"""
		self._logger_class = logger_class
		self._logger_name = logger_name

	def __enter__(self):
		"""
		Store logging.getLoggerClass()
		"""
		self._orig_logger_class = logging.getLoggerClass()
		if self._logger_class is not None:
			logging.setLoggerClass(self._logger_class)
		if self._logger_name is not None:
			return logging.getLogger(self._logger_name)
		else:
			return None

	def __exit__(self, exc_type, exc_value, traceback):
		"""
		Restore the logging.getLoggerClass() with logging.setLoggerClass().
		"""
		logging.setLoggerClass(self._orig_logger_class)

def get_logger_with_class_if_new(logger_class, logger_name, *args, set_default_logger_class=False, **kwargs):
	"""
	If the logger name is new, return a new logger of this class.  Call
	`set_default_logger_class` to keep this class as the default logger class
	with `logging.setLoggerClass()`.
	"""
	if set_default_logger_class:
		logging.setLoggingClass(logger_class)
		return logging.getLogger(logger_name)
	else:
		with PushDefaultLoggingClassContext(logger_class=logger_class, logger_name=logger_name) as logger:
			return logger

class ConsoleFormatter(logging.Formatter):
	"""
	Let users write e.g. logger.info("message", extra=dict(color='white'))

	Currently, only "white" is supported.
	"""

	def format(self, record):
		"""
		Let users write e.g. logger.info("message", extra=dict(color='white'))

		Currently, only "white" is supported.
		"""

		# Use copy.copy - c.f. https://stackoverflow.com/a/7961390
		colored_record = copy.copy(record)

		color = None
		try:
			color = record.color
		except AttributeError as e:
			pass

		if color is not None:
			if color is None or not color or color == "none":
				pass
			elif color == "white":
				white = "\033[37m"
				clear = "\033[0;0m"
				colored_record.msg = "{0:s}{1:s}{2:s}".format(
					white,
					colored_record.msg,
					clear,
				)
			else:
				raise WCMIError("error: ConsoleFilter: unrecognized color `{0:s}'.".format(str(color)))

		return super().format(colored_record)

default_console_level = logging.INFO
default_formatter = logging.Formatter(
	"%(message)s",
)
console_formatter = ConsoleFormatter(
	"%(message)s",
)

# From https://docs.python.org/3/howto/logging-cookbook.html.
timestamped_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

stdout_handler = logging.StreamHandler(sys.stdout)
stderr_handler = logging.StreamHandler(sys.stderr)

class StdoutFilter(logging.Filter):
	"""
	Ignore messages that stderr handles: hide levels >= logging.WARNING.
	"""

	def filter(self, record):
		"""
		Ignore messages that stderr handles: hide levels >= logging.WARNING.
		"""

		if record.levelno >= logging.WARNING:
			return False

		return True

stdout_filter = StdoutFilter()

# Set logger level to handle all messages, and let the handlers and
# filters decide what to handle.

# When logger_level is logging.NOTSET, nothing seems to be printed at all.
logger_level = max(logging.NOTSET + 1, min(logging.NOTSET + 1, logging.DEBUG, default_console_level))

# c.f. https://stackoverflow.com/a/7961390
class Logger(*list(dict.fromkeys((logging.getLoggerClass(), logging.Logger,)))):
	"""
	Forward e.g. logging.info("message", color="white") as logging.info("message", extra={"color": "white"})

	Note: these logging methods in my Python 3.7 implementation call `_log()`.
	"""
	def log_forward_color(self, forward_func, *args, color=None, **kwargs):
		"""
		Process arguments to forward to log methods.
		"""
		if color is None:
			return forward_func(*args, **kwargs)
		else:
			if "extra" not in kwargs:
				return forward_func(*args, extra=dict(color=color), **kwargs)
			else:
				return forward_func(*args, **{**kwargs, "extra": {**kwargs["extra"], "color": color}})

	def log(self, level, msg, color=None, *args, **kwargs):
		"""
		Forward e.g. logging.log(logging.INFO, "message", color="white") as logging.log(logging.INFO, "message", extra={"color": "white"})
		"""
		return self.log_forward_color(super().log, level, msg, *args, color=color, **kwargs)

	def critical(self, msg, color=None, *args, **kwargs):
		"""
		Forward e.g. logging.critical("message", color="white") as logging.critical("message", extra={"color": "white"})
		"""
		return self.log_forward_color(super().critical, msg, *args, color=color, **kwargs)

	def error(self, msg, color=None, *args, **kwargs):
		"""
		Forward e.g. logging.error("message", color="white") as logging.error("message", extra={"color": "white"})
		"""
		return self.log_forward_color(super().error, msg, *args, color=color, **kwargs)

	def warning(self, msg, color=None, *args, **kwargs):
		"""
		Forward e.g. logging.warning("message", color="white") as logging.warning("message", extra={"color": "white"})
		"""
		return self.log_forward_color(super().warning, msg, *args, color=color, **kwargs)

	def info(self, msg, color=None, *args, **kwargs):
		"""
		Forward e.g. logging.info("message", color="white") as logging.info("message", extra={"color": "white"})
		"""
		return self.log_forward_color(super().info, msg, *args, color=color, **kwargs)

	def debug(self, msg, color=None, *args, **kwargs):
		"""
		Forward e.g. logging.debug("message", color="white") as logging.debug("message", extra={"color": "white"})
		"""
		return self.log_forward_color(super().debug, msg, *args, color=color, **kwargs)

logger_class = Logger

logger = get_logger_with_class_if_new(logger_class, logger_name)

def handler_defaults(stdout_handler=stdout_handler, stderr_handler=stderr_handler):
	"""
	Reset the logger handlers to the default configuration specified in this
	module, leaving unspecified configuration unchanged.
	"""

	for handler in [stdout_handler, stderr_handler]:
		if handler is not None:
			handler.setFormatter(default_formatter)

	if stdout_handler is not None:
		stdout_handler.setLevel(default_console_level)
		# Ignore messages given to stderr.
		stdout_handler.addFilter(stdout_filter)

	if stderr_handler is not None:
		stderr_handler.setLevel(max(default_console_level + 1, logging.WARNING))

def logger_defaults(logger=logger, stdout_handler=stdout_handler, stderr_handler=stderr_handler):
	"""
	Reset the logger to the default configuration specified in this module,
	leaving unspecified configuration unchanged.
	"""

	# Reset the handlers.
	handler_defaults(stdout_handler=stdout_handler, stderr_handler=stderr_handler)

	# Skip if logger is unspecified.
	if logger is not None:
		## Set the logger class.
		#if logger_class is not None:
		#	logger.setClass(logger_class)

		# Add handlers.
		for handler in [stdout_handler, stderr_handler]:
			if handler is not None:
				logger.addHandler(handler)

		# Set logger level to handle all messages, and let the handlers and
		# filters decide what to handle.
		logger.setLevel(logger_level)

logger_defaults()

def verbose(verbosity=1, stdout_handler=stdout_handler, logger=logger):
	"""
	Also print DEBUG-level messages to standard output.
	"""
	if verbosity >= 1:
		if stdout_handler is not None:
			stdout_handler.setLevel(min(default_console_level, logging.DEBUG))
	else:
		if stdout_handler is not None:
			stdout_handler.setLevel(default_console_level)
