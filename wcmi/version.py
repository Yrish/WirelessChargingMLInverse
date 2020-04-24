# -*- coding: utf-8 -*-
# vim: set noet ft=python :

version = (0,1,0)

def version_to_str(ver=None):
	"""
	Given e.g. (0,1,0), return "0.1.0".
	"""
	if ver is None:
		ver = version
	return ".".join(str(component) for component in ver)

version_str = version_to_str(version)

def version_compatible(other, ours=None, num_components=1):
	"""
	Determine whether the first num_components of the versions are equal.
	"""
	if ours is None:
		ours = version

	if (other + num_components*(0,))[:num_components] != (ours + num_components*(0,))[:num_components]:
		return False

	if (len(other) >= 1) != (len(ours) >= 1):
		return False

	return True
