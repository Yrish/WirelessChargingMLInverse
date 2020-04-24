# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
wcmi package: WirelessChargingMLInverse
"""

from wcmi.cli import main
from wcmi.version import version_str

__version__ = version_str

__all__ = [
	'main',
]

if __name__ == "__main__":
	import sys
	main(sys.argv[:])
