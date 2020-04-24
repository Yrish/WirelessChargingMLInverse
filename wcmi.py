#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: set noet ft=python :

"""
Simple wrapper around wcmi.main.
"""

import sys, os, os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import wcmi.cli

if __name__ == "__main__":
	import sys
	wcmi.cli.main(sys.argv[:])
