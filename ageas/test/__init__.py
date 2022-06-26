#!/usr/bin/env python3
"""
Test script to make sure AGEAS working

author: jy, nkmtmsys
"""
import ageas
from pkg_resources import resource_filename
group1_path = resource_filename(__name__, 'ips.csv.gz')
group2_path = resource_filename(__name__, 'mef.csv.gz')



def Test(cpu_mode:bool = False):
	"""
	Automatically select device if cpu_mode is on,
	AGEAS will be forced to only use CPU

	Args:
		cpu_mode: <bool> Default = False
			If cpu_mode is on, AGEAS will be forced to only use CPU.
			Otherwise, AGEAS will automatically select device, preferring GPUs.

	Outputs:
		ageas._main.Launch object

	Example::
		import ageas
		result = ageas.Test(cpu_mode = False)
	"""
	print('Start Test')
	easy = ageas.Launch(
		cpu_mode = cpu_mode,
		group1_path = group1_path,
		group2_path = group2_path,
		protocol = 'multi',
	)
	assert 'Nanog' in easy.atlas.regulons['regulon_0'].genes
	assert 'Klf4' in easy.atlas.regulons['regulon_0'].genes
	print('Finished Test. LGTM')
	return easy
