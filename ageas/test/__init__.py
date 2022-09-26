#!/usr/bin/env python3
"""
Test script to make sure AGEAS working

author: jy, nkmtmsys
"""

import ageas
from pkg_resources import resource_filename
class1_path = resource_filename(__name__, 'ips.csv.gz')
class2_path = resource_filename(__name__, 'mef.csv.gz')



def Test(**kwargs):
	"""
	Function to test whether AGEAS is performing properly or not.

	:param kwargs: All args in ageas.Launch except **class1_path**,
		**class2_path**, **sliding_window_size**

	:return: <ageas._main.Launch object>
	"""
	print('Start Test')
	easy = ageas.Launch(
		class1_path = class1_path,
		class2_path = class2_path,
		sliding_window_size = 10,
		**kwargs
		# meta_load_path = resource_filename(__name__, 'metaGRN.js'),
		# psgrn_load_path = resource_filename(__name__, 'psGRNs.js'),
	)
	assert 'Nanog' in easy.atlas.nets[0].genes
	assert 'Klf4' in easy.atlas.nets[0].genes
	print('Finished Test. LGTM')
	return easy

# little example to save report files
# if __name__ == '__main__':
# 	test = ageas.Test(
# 		cpu_mode = True,
# 		# protocol = 'multi',
# 		mute_unit = False,
# 		normalize = 'Min_Max_1000',
# 	)
# 	test.save_reports(
# 		folder_path = 'easy_test.report',
# 		save_unit_reports = True
# 	)
# 	print('CMPL Easy Test')
