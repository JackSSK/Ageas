#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import pandas as pd
import ageas.lib as lib
import ageas.tool as tool
from collections import OrderedDict


GRP_TYPES = ['Standard', 'Outer_Signifcant', 'Outer_Supportive']


class Extract(object):
	"""
	Extract key genes from the most important GRPs
	"""

	def __init__(self,
				grp_importances = None,
				score_thread = None,
				outlier_grps = {},
				top_grp_amount = None):
		super(Extract, self).__init__()
		self.regulons = []
		self.key_genes = None
		self.common_reg_source = None
		self.common_reg_target = None
		self.outlier_grps = outlier_grps
		self.grps = grp_importances.stratify(score_thread,
											top_grp_amount,
											len(outlier_grps))

	# as named
	def construct_regulon(self, meta_grn):
		self.key_genes = self.__extract_genes(self.grps, self.outlier_grps)
		# process standard grps
		for id in self.grps.index:
			try:
				grp = meta_grn[id]
			except Exception:
				raise lib.Error('GRP', id, 'not in GRN Guide')
			grp['type'] = GRP_TYPES[0]
			grp['score'] = self.grps.loc[id]['importance']
			self.__update_regulon_with_grp(grp)
		for id in self.outlier_grps:
			try:
				grp = meta_grn[id]
			except Exception:
				raise lib.Error('GRP', id, 'not in GRN Guide')
			grp['type'] = GRP_TYPES[1]
			grp['score'] = self.outlier_grps[id]
			self.__update_regulon_with_grp(grp)

	# as named
	def __update_regulon_with_grp(self, grp):
		source = grp['regulatory_source']
		target = grp['regulatory_target']
		start_new = True
		# check whether GRP could be appended into an exist regulon
		for regulon in self.regulons:
			if source in regulon['genes'] or target in regulon['genes']:
				start_new = False
				# append GRP into regulon
				assert grp['id'] not in regulon['grps']
				regulon['grps'][grp['id']] = grp
				# update gene list
				if source not in regulon['genes']:
					regulon['genes'][source] = {'source':[], 'target':[]}
				elif target not in regulon['genes']:
					regulon['genes'][target] = {'source':[], 'target':[]}

				assert source not in regulon['genes'][target]['source']
				assert source not in regulon['genes'][target]['target']
				assert target not in regulon['genes'][source]['source']
				assert target not in regulon['genes'][source]['target']
				regulon['genes'][target]['source'].append(source)
				regulon['genes'][source]['target'].append(target)
				if grp['reversable']:
					regulon['genes'][source]['source'].append(target)
					regulon['genes'][target]['target'].append(source)
				break
		# make new regulon data
		if start_new:
			regulon = {
				'grps':{grp['id']:grp},
				'genes':{
					source:{'source': [], 'target': [target]},
					target:{'source': [source], 'target': []}
				}
			}
			if grp['reversable']:
				regulon['genes'][source]['source'].append(target)
				regulon['genes'][target]['target'].append(source)
			self.regulons.append(regulon)

	# extract common regulaoty sources or targets of given genes
	def extract_common(self,
						meta_grn,
						type = 'regulatory_source',
						occurrence_thread = 1):
		if type == 'regulatory_source': known = 'regulatory_target'
		elif type == 'regulatory_target': known = 'regulatory_source'
		genes = [x[0] for x in self.key_genes]
		dict = {}
		for grp in meta_grn:
			record = meta_grn[grp]
			if record[known] in genes:
				target = record[type]
				if target not in dict:
					dict[target] = {
						'relate': [{record[known]:self._new_rec(record)}],
						'influence': 1
					}
				else:
					dict[target]['relate'].append(
										{record[known]:self._new_rec(record)})
					dict[target]['influence'] += 1
		dict = {ele:dict[ele]
				for ele in dict
					if dict[ele]['influence'] >= occurrence_thread}
		dict = OrderedDict(sorted(dict.items(),
									key = lambda x: x[1]['influence'],
									reverse = True))
		if type == 'regulatory_source':      self.common_reg_source = dict
		elif type == 'regulatory_target':    self.common_reg_target = dict

	# find factors by checking Ageas' assigned importancy and regulaotry impact
	def report(self):
		factors = {k[0]:k[1] for k in self.key_genes}
		temp = {}
		for ele in self.common_reg_source:
			reg_target_num = self.common_reg_source[ele]['influence']
			if ele in factors:
				temp[ele] = [factors[ele], reg_target_num]
			else:
				temp[ele] = [[0, 0], reg_target_num]
		# for ele in factors:
		# 	if ele not in temp: temp[ele] = [factors[ele], 0]
		temp = [[
					k, temp[k][0][0], temp[k][1], temp[k][0][1]
				]for k in temp if max(temp[k][1], temp[k][0][1]) >= 2]
		temp = sorted(temp, key = lambda x: x[-1], reverse = True)
		temp = pd.DataFrame(temp, columns = ['Gene','Score','Degree','Count'])
		temp['Score'] = temp['Score'] / temp['Count']
		return temp

	# extract genes based on whether occurence in important GRPs passing thread
	def __extract_genes(self, stratified_grps, outlier_grps):
		dict = {}
		for ele in stratified_grps.index.tolist():
			score = stratified_grps.loc[ele]['importance']
			self.__update_gene_extract_dict(ele, score, dict)
		for ele in outlier_grps:
			self.__update_gene_extract_dict(ele, outlier_grps[ele], dict)
		# filter by top_grp_amount
		answer = [[e, dict[e]] for e in dict]
		answer.sort(key = lambda x:x[-1][-1], reverse = True)
		return answer

	def __update_gene_extract_dict(self, grp, score, dict):
		grp = grp.strip().split('_') # get source and target from GRP ID
		if grp[0] not in dict:
			dict[grp[0]] = [score, 1]
		else:
			dict[grp[0]][0] += score
			dict[grp[0]][1] += 1
		if grp[1] not in dict:
			dict[grp[1]] = [score, 1]
		else:
			dict[grp[1]][0] += score
			dict[grp[1]][1] += 1
		return

	# Add correlation in class 1 and class 2 into regulon record
	def _new_rec(self, rec):
		return {k:rec[k] for k in rec if k not in ['id',
													'regulatory_source',
													'regulatory_target']}
