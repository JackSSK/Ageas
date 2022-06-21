#!/usr/bin/env python3
"""
Test script to make sure AGEAS working

author: jy, nkmtmsys
"""
import ageas
from pkg_resources import resource_filename
group1_path = resource_filename(__name__, 'ips.csv')
group2_path = resource_filename(__name__, 'mef.csv')


# Automatically select device
# If cpu_mode is on, AGEAS will be forced to only use CPU
def Test(cpu_mode = False):
	print('Start Test')
	easy = ageas.Launch(
		cpu_mode = cpu_mode,
		group1_path = group1_path,
		group2_path = group2_path,
		protocol = 'multi',
	)
	print('Finished Test. LGTM')
