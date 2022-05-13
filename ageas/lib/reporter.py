#!/usr/bin/env python3
"""
Ageas Reborn

ToDo:
Need to be generalized

author: jy, nkmtmsys
"""

import collections
import pandas as pd
import ageas.tool.json as json

# find factors simply by GRN degree
def grn_degree_based_analysis(path, top = 50):
	grn_abundance = {}
	grn = json.decode(path)
	for ele in grn:
		source = grn[ele]['regulatory_source']
		target = grn[ele]['regulatory_target']
		if source not in grn_abundance: grn_abundance[source] = 1
		else: grn_abundance[source] += 1
		if target not in grn_abundance: grn_abundance[target] = 1
		else: grn_abundance[target] += 1
	return collections.Counter(grn_abundance).most_common(top)

# find factors by checking Ageas' assigned importancy and regulaotry impact
def ageas_based_analysis(commonSource, factors):
	common = json.decode(commonSource)
	factors = json.decode(factors)
	factors = {k[0]:k[1] for k in factors}
	result = {}
	for ele in common:
		if ele in factors:
			result[ele] = [factors[ele], common[ele]['influence']]
		else: result[ele] = [0, common[ele]['influence']]
	# for ele in factors:
	# 	if ele not in result: result[ele] = [factors[ele], 0]
	result = [[k, result[k][0], result[k][1]] for k in result]
	result = sorted(result, key = lambda x: x[1], reverse = True)
	return result

# overall analysis script
def analysis(folder):
	# get ageas based result
	candidates = ageas_based_analysis(folder + 'common_source.js',
                                        folder +'repeated_factors.js')
	candidates = pd.DataFrame(candidates, columns = ['Gene', 'Score', 'Degree'])
	candidates.to_csv(folder + 'ageas_based.csv', index = False )
	# get GRN degree based result
	grn_abundance = grn_degree_based_analysis(folder + 'grn_guide.js',
                                                top = len(candidates.index))
	grn_abundance = [[k[0],k[1]] for k in grn_abundance]
	grn_abundance = pd.DataFrame(grn_abundance,
                                columns = ['Gene', 'Degree'])
	grn_abundance.to_csv(folder + 'grn_based.csv',
                        index = False )
