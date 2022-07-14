#!/usr/bin/env python3
"""
Test script to make sure AGEAS working

author: jy, nkmtmsys
"""

import ageas
from pkg_resources import resource_filename
group1_path = resource_filename(__name__, 'ips.csv.gz')
group2_path = resource_filename(__name__, 'mef.csv.gz')



def Test(**kwargs):
	"""
	Function to test whether AGEAS is performing properly or not.

	Parameters:
		kwargs: All args in ageas.Launch except **group1_path**,
		**group2_path**, **sliding_window_size**

	Outputs:
		ageas._main.Launch object

	"""
	print('Start Test')
	easy = ageas.Launch(
		group1_path = group1_path,
		group2_path = group2_path,
		sliding_window_size = 10,
		**kwargs
		# meta_load_path = resource_filename(__name__, 'metaGRN.js'),
		# psgrn_load_path = resource_filename(__name__, 'psGRNs.js'),
	)
	assert 'Nanog' in easy.atlas.regulons[0].genes
	assert 'Klf4' in easy.atlas.regulons[0].genes
	print('Finished Test. LGTM')
	return easy
