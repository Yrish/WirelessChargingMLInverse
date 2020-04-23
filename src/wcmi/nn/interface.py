# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
Module providing methods to interface with the neural networks provided by this
package.
"""

import wcmi.nn.gan as gan

def train(use_gan=True, load_model=None, save_model=None, load_data=None, gan_n=gan.default_gan_n):
	"""
	(To be documented...)
	"""
	if gan_n is None:
		gan_n = gan.default_gan_n
	print("(To be implemented...)")
	pass

def run(use_gan=True, load_model=None, save_model=None, load_data=None, save_data=None, gan_n=gan.default_gan_n):
	"""
	(To be documented...)
	"""
	if gan_n is None:
		gan_n = gan.default_gan_n
	print("(To be implemented...)")
	pass

def stats():
	"""
	(To be documented...)
	"""
	print("(To be implemented...)")
	pass
