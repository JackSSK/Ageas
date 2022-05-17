#!/usr/bin/env python3
"""
Ageas Reborn

author: jy, nkmtmsys
"""

import pandas as pd
import ageas.lib as lib
import ageas.tool as tool
from collections import OrderedDict


GRP_TYPES = ['Standard', 'Outer_Signifcant', 'Outer_Bridge']


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
	def construct_regulon(self, meta_grn, header = 'regulon_'):
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

		# combine regulons if sharing common genes
		i = 0
		j = 1
		while True:
			if i == len(self.regulons) or j == len(self.regulons): break
			reg_1 = self.regulons[i]
			reg_2 = self.regulons[j]
			if len([ele for ele in reg_1['genes'] if ele in reg_2['genes']])>0:
				self.__update_regulon_with_another(regulon = reg_1, res = reg_2)
				del self.regulons[j]
				j = i + 1
			else:
				j += 1
				if j == len(self.regulons):
					i += 1
					j = i + 1
		# change regulon to dict type and find key genes
		self.regulons = {header + str(i):e for i, e in enumerate(self.regulons)}
		self.key_genes = self.__extract_genes(self.grps, self.outlier_grps)

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

	# as named
	def __update_regulon_with_another(self, regulon, res):
		regulon['grps'].update(res['grps'])
		for gene in res['genes']:
			if gene in regulon['genes']:
				regulon['genes'][gene]['source'].extend(
										res['genes'][gene]['source'])
				regulon['genes'][gene]['target'].extend(
										res['genes'][gene]['target'])
			else:
				regulon['genes'][gene] = res['genes'][gene]

	# extract common regulaoty sources or targets of given genes
	def extract_common(self,
						meta_grn,
						type = 'regulatory_source',
						occurrence_thread = 1):
		if type == 'regulatory_source': known = 'regulatory_target'
		elif type == 'regulatory_target': known = 'regulatory_source'
		genes = {x[0]:None for x in self.key_genes}
		dict = {}
		for grp in meta_grn:
			record = meta_grn[grp]
			if record[known] in genes:
				target = record[type]
				if target not in dict:
					dict[target] = {
						'relate': [{record[known]:self.__copy_rec(record)}],
						'influence': 1
					}
				else:
					dict[target]['relate'].append(
										{record[known]:self.__copy_rec(record)})
					dict[target]['influence'] += 1
		dict = {ele:dict[ele] for ele in dict
								if dict[ele]['influence'] >= occurrence_thread}
		dict = OrderedDict(sorted(dict.items(),
								key = lambda x:x[1]['influence'], reverse=True))
		if type == 'regulatory_source':      self.common_reg_source = dict
		elif type == 'regulatory_target':    self.common_reg_target = dict

	# find factors by checking Ageas' assigned importancy and regulaotry impact
	def report(self):
		factors = {k[0]:k[1] for k in self.key_genes}
		for ele in self.common_reg_source:
			reg_target_num = self.common_reg_source[ele]['influence']
			if ele in factors: 	factors[ele].append(reg_target_num)
			else:				factors[ele] = ['None', 0, 0, reg_target_num]
		for ele in factors:
			if len(factors[ele]) < 4: factors[ele].append(0)
		factors = [[k,factors[k][0],factors[k][1],factors[k][2],factors[k][3]]
					for k in factors if max(factors[k][2], factors[k][3]) >= 2]
		factors = pd.DataFrame(sorted(factors,key=lambda x:x[-2],reverse=True))
		factors.columns=['Gene','Regulon','Source_Num','Target_Num','Influence']
		return factors

	# extract genes based on whether occurence in important GRPs passing thread
	def __extract_genes(self, stratified_grps, outlier_grps):
		dict = {}
		for regulon_id in self.regulons:
			regulon = self.regulons[regulon_id]
			for gene in regulon['genes']:
				source_num = len(regulon['genes'][gene]['source'])
				target_num = len(regulon['genes'][gene]['target'])
				if gene not in dict:
					dict[gene] = [regulon_id, source_num, target_num]
				else: raise lib.Error('Repeated Gene in regulons', gene)
		# filter by top_grp_amount
		answer = [[e, dict[e]] for e in dict]
		answer.sort(key = lambda x:x[-1][-1], reverse = True)
		return answer


	# Add correlation in class 1 and class 2 into regulon record
	def __copy_rec(self, rec):
		return {k:rec[k] for k in rec if k not in ['id',
													'regulatory_source',
													'regulatory_target']}
