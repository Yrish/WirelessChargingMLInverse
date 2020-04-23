# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
wcmi package: WirelessChargingMLInverse
"""

__version__ = '0.1.0'

from wcmi.cli import main

__all__ = [
	'main',
]

if __name__ == "__main__":
	import sys
	main(sys.argv[:])
